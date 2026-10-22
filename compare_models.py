import os
import json
import matplotlib.pyplot as plt
import glob
import numpy as np

def find_latest_run(model_name):
    runs = glob.glob(f"runs/{model_name}*")
    if not runs:
        return None
    runs.sort(key=os.path.getmtime, reverse=True)
    return runs[0]

def extract_metrics(run_dir):
    try:
        with open(os.path.join(run_dir, "results.json"), 'r') as f:
            data = json.load(f)
        return data['results']['history']
    except Exception as e:
        print(f"Hata: {run_dir} icinde results.json okunamadi. {e}")
        return None

def analyze_history(history, model_name):
    val_loss = history['val_loss']
    train_loss = history['train_loss']
    val_auc = history['val_auc']
    
    min_val_loss_epoch = val_loss.index(min(val_loss))
    max_auc_epoch = val_auc.index(max(val_auc))
    
    print(f"\n--- {model_name.upper()} Analizi ---")
    print(f"En Dusuk Validation Loss: Epoch {min_val_loss_epoch + 1} ({min(val_loss):.4f})")
    print(f"En Yuksek Validation AUC: Epoch {max_auc_epoch + 1} ({max(val_auc):.4f})")
    
    # Overfitting Tespiti: Val loss'un min oldugu noktadan sonra artmaya baslamasi
    if min_val_loss_epoch < len(val_loss) - 1:
        print(f"⚠️ DİKKAT: Model {min_val_loss_epoch + 1}. epoch'tan sonra ezberlemeye (overfitting) baslamis olabilir.")
        print(f"Cunku bu noktadan sonra validation loss artisa gecerken train loss dusmeye devam ediyor.")
    else:
        print("Model egitim boyunca belirgin bir sekilde ezberlemeye baslamamis.")
        
    print(f"Tavsiye Edilen Optimum Epoch: {max_auc_epoch + 1} (En Yuksek Basari)")

def plot_comparison(hist1, name1, hist2, name2):
    epochs = range(1, len(hist1['train_loss']) + 1)
    
    plt.figure(figsize=(18, 6))
    
    # Train Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, hist1['train_loss'], label=f'{name1} Train', linestyle='-')
    if hist2:
        plt.plot(epochs, hist2['train_loss'], label=f'{name2} Train', linestyle='--')
    plt.title('Egitim Kaybi (Train Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Val Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, hist1['val_loss'], label=f'{name1} Val', linestyle='-')
    if hist2:
        plt.plot(epochs, hist2['val_loss'], label=f'{name2} Val', linestyle='--')
    plt.title('Dogrulama Kaybi (Val Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Val AUC
    plt.subplot(1, 3, 3)
    plt.plot(epochs, hist1['val_auc'], label=f'{name1} Val AUC', marker='o', markersize=4)
    if hist2:
        plt.plot(epochs, hist2['val_auc'], label=f'{name2} Val AUC', marker='s', markersize=4)
    plt.title('Dogrulama Basarisi (Val AUC)')
    plt.xlabel('Epochs')
    plt.ylabel('AUC Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_karsilastirma.png', dpi=300)
    print("\nKarsilastirma grafikleri 'model_karsilastirma.png' olarak kaydedildi.")
    plt.close()

def main():
    eff_run = find_latest_run("efficientnet")
    res_run = find_latest_run("resnet")
    
    if not eff_run and not res_run:
        print("Hic egitim sonucu bulunamadi. Lutfen once 'python main.py' calistirin.")
        return
        
    hist_eff, hist_res = None, None
    
    if eff_run:
        print(f"EfficientNet sonuclari aliniyor: {eff_run}")
        hist_eff = extract_metrics(eff_run)
        if hist_eff: analyze_history(hist_eff, "EfficientNet-V2-S")
        
    if res_run:
        print(f"ResNet sonuclari aliniyor: {res_run}")
        hist_res = extract_metrics(res_run)
        if hist_res: analyze_history(hist_res, "ResNet-50")
        
    if hist_eff and hist_res:
        plot_comparison(hist_eff, "EfficientNet", hist_res, "ResNet")
    elif hist_eff:
        plot_comparison(hist_eff, "EfficientNet", None, None)
    elif hist_res:
        plot_comparison(hist_res, "ResNet", None, None)

if __name__ == "__main__":
    main()
