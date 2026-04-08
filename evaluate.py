import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import get_model
import numpy as np
import os

def evaluate_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Cihaz: {device}")

    print("Dogrulama verileri yukleniyor...")
    _, val_loader, _, val_df = get_dataloaders(
        csv_path='dataset/train.csv', 
        img_dir='dataset/train', 
        batch_size=32, 
        img_size=224
    )
    
    model = get_model(pretrained=False)
    if not os.path.exists("best_model.pth"): return
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    print("Tahminler hesaplaniyor...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            if probs.ndim == 0: probs = np.array([probs])
                
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Egrisi (AUC = %{roc_auc*100:.1f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlis Alarm Orani (False Positive Rate)')
    plt.ylabel('Kanseri Yakalama Orani (True Positive Rate)')
    
    best_preds = (np.array(all_probs) >= 0.50).astype(int)
    best_acc = accuracy_score(all_labels, best_preds) * 100
    plt.title(f'ROC Egrisi - %{best_acc:.1f} Dogruluk')
    
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    thresholds_to_test = [0.5, 0.75, 0.85]
    
    for thresh in thresholds_to_test:
        preds = (np.array(all_probs) >= thresh).astype(int)
        cm = confusion_matrix(all_labels, preds)
        accuracy_val = accuracy_score(all_labels, preds) * 100
        
        fp = cm[0,1]
        tp = cm[1,1]
        fn = cm[1,0]
        tn = cm[0,0]
        print(f"Esik %{int(thresh*100)} => Dogruluk: %{accuracy_val:.2f} | TP: {tp} | FP: {fp} | FN: {fn}")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=["Tahmin: Iyi", "Tahmin: Kanser"], 
                    yticklabels=["Gercek: Iyi", "Gercek: Kanser"])
        plt.title(f"Karisiklik Matrisi (Esik: %{int(thresh*100)}) | Dogruluk: %{accuracy_val:.1f}")
        plt.ylabel('Gercek Degerler')
        plt.xlabel('Tahmin Edilen Degerler')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{int(thresh*100)}.png')
        plt.close()

    print("Degerlendirme tamamlandi:")
    print("-> roc_curve.png")
    print("-> confusion_matrix_50.png")
    print("-> confusion_matrix_75.png")
    print("-> confusion_matrix_85.png")

if __name__ == "__main__":
    evaluate_model()
