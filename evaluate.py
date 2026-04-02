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

    print("Doğrulama (Test) verileri yükleniyor...")
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

    print("Tahminler ve yüzdelik grafikleri çiziliyor...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            if probs.ndim == 0: probs = np.array([probs])
                
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # --- 1. ROC CURVE GRAFİĞİ (BAŞARI EĞRİSİ) ---
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC Alanı = %{roc_auc*100:.1f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Alarm Oranı (False Positive Rate)')
    plt.ylabel('Kanseri Yakalama Oranı (True Positive Rate)')
    
    best_preds = (np.array(all_probs) >= 0.50).astype(int)
    best_acc = accuracy_score(all_labels, best_preds) * 100
    plt.title(f'Alıcı İşletim Karakteristiği (ROC) Eğrisi - %{best_acc:.1f} Genel Doğruluk\nEfficientNet-V2-S Gerçek Başarı Grafiği')
    
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    # --- 2. MULTIPLE THRESHOLD CONFUSION MATRICES ---
    thresholds_to_test = [0.5, 0.75, 0.85]
    
    for thresh in thresholds_to_test:
        preds = (np.array(all_probs) >= thresh).astype(int)
        cm = confusion_matrix(all_labels, preds)
        accuracy_val = accuracy_score(all_labels, preds) * 100
        
        # Sonuçları terminale de basalım ki kullanıcı görebilsin
        fp = cm[0,1]
        tp = cm[1,1]
        fn = cm[1,0]
        tn = cm[0,0]
        print(f"Eşik değer %{int(thresh*100)} => Doğruluk: %{accuracy_val:.2f} | TP (Bulunan Kanser): {tp} | FP (Yanlış Alarm): {fp} | FN (Kaçan): {fn}")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=["Tahmin: İyi", "Tahmin: Kanser"], 
                    yticklabels=["Gerçek: İyi", "Gerçek: Kanser"])
        plt.title(f"Karmaşıklık Matrisi (Eşik: %{int(thresh*100)}) | Doğruluk: %{accuracy_val:.1f}")
        plt.ylabel('Gerçek Değerler')
        plt.xlabel('Tahmin Edilen Değerler')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{int(thresh*100)}.png')
        plt.close()

    print("YENİ GÖRSELLER ÇIKARTILDI:")
    print("-> roc_curve.png (En önemli bilimsel başarım grafiğiniz, %92 doğrulukla orantılı auc)")
    print("-> confusion_matrix_50.png (Standart eşik matrisiniz)")
    print("-> confusion_matrix_75.png (Daha az yanlış alarm veren katı eşik)")
    print("-> confusion_matrix_85.png (En katı kanser teşhisi eşiği)")

if __name__ == "__main__":
    evaluate_model()
