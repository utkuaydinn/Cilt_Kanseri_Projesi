import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import copy
import matplotlib.pyplot as plt
import time

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5, scheduler=None):
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_auc = 0.0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'val_auc': [], 'val_recall': [], 'val_precision': [],
        'epoch_times': []
    }
    
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            all_labels = []
            all_preds = []      
            all_preds_bin = []  

            loop = tqdm(dataloader, desc=f"{phase.capitalize()} Phase")
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                preds_bin = (probs >= 0.5).astype(int)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs)
                all_preds_bin.extend(preds_bin)
                
                loop.set_postfix(loss=loss.item())

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            
            if len(np.unique(all_labels)) > 1:
                epoch_auc = roc_auc_score(all_labels, all_preds)
            else:
                epoch_auc = 0.0
                
            epoch_recall = recall_score(all_labels, all_preds_bin, zero_division=0)
            epoch_precision = precision_score(all_labels, all_preds_bin, zero_division=0)
            epoch_f1 = f1_score(all_labels, all_preds_bin, zero_division=0)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} | '
                  f'AUC: {epoch_auc:.4f} | Recall: {epoch_recall:.4f} | Precision: {epoch_precision:.4f} | F1: {epoch_f1:.4f}')
                  
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_auc'].append(epoch_auc)
                history['val_recall'].append(epoch_recall)
                history['val_precision'].append(epoch_precision)

            if phase == 'val' and epoch_auc > best_val_auc:
                best_val_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        history['epoch_times'].append(epoch_duration)
        print(f'Epoch Süresi: {epoch_duration:.2f}s\n')

    total_time = time.time() - total_start_time
    print(f'Eğitim Tamamlandı! Toplam Süre: {total_time//60:.0f}m {total_time%60:.0f}s')
    print(f'En İyi Validation AUC: {best_val_auc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

def plot_history(history, save_path='training_history.png'):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_auc'], label='Val AUC', marker='o')
    plt.plot(epochs, history['val_recall'], label='Val Recall', marker='s')
    plt.title('Validation Metrics (AUC, Recall)')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Eğitim grafikleri '{save_path}' olarak diske kaydedildi.")
    plt.close()

