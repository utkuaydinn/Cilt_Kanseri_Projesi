import torch
import torch.optim as optim
import torch.nn as nn
import os
import argparse
import json
from datetime import datetime
from dataset import get_dataloaders
from model import get_model
from trainer import train_model, plot_history
from utils import FocalLoss

def main():
    # 1. Komut Satırı Argümanları
    parser = argparse.ArgumentParser(description='Cilt Kanseri Tespiti Eğitim Scripti')
    parser.add_argument('--model', type=str, default='efficientnet', choices=['efficientnet', 'resnet'], 
                        help='Eğitilecek model mimarisi')
    parser.add_argument('--batch', type=int, default=32, help='Batch size değeri')
    parser.add_argument('--epochs', type=int, default=5, help='Epoch sayısı')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_argument_list if hasattr(parser, 'parse_argument_list') else parser.parse_args()

    # 2. Cihaz Ayarı (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Sistem: NVIDIA GPU (CUDA) algılandı. Cihaz: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Sistem: Apple Silicon (MPS) algılandı.")
    else:
        device = torch.device("cpu")
        print("Sistem: GPU bulunamadı, CPU kullanılıyor.")

    # 3. Klasörleme ve Kayıt Hazırlığı
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_B{args.batch}_{timestamp}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Bu eğitim '{run_dir}' klasörüne kaydedilecek.")

    # Hiperparametreler
    csv_path = 'dataset/train.csv'
    img_dir = 'dataset/train'
    img_size = 224

    # 4. Dataloader'ları Yükle
    print(f"Veri yükleniyor (Batch: {args.batch}, Size: {img_size})...")
    train_loader, val_loader, train_df, val_df = get_dataloaders(
        csv_path=csv_path, img_dir=img_dir, batch_size=args.batch, img_size=img_size
    )

    # 5. Modeli Yükle
    model = get_model(model_name=args.model, pretrained=True)

    # 6. Kayıp Fonksiyonu ve Optimizer
    criterion = FocalLoss(alpha=0.85, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 7. Eğitimi Başlat
    print(f"Eğitim Başlıyor: {args.model} | Epochs: {args.epochs}")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs
    )

    # 8. Sonuçları Kaydet
    # Model Ağırlıkları
    model_save_path = os.path.join(run_dir, "best_model.pth")
    torch.save(trained_model.state_dict(), model_save_path)
    
    # Grafikler
    plot_save_path = os.path.join(run_dir, "training_history.png")
    plot_history(history, save_path=plot_save_path)
    
    # Metrikler ve Konfigürasyon (JSON)
    results = {
        'config': vars(args),
        'results': {
            'best_val_auc': max(history['val_auc']) if history['val_auc'] else 0,
            'total_time_seconds': sum(history['epoch_times']),
            'history': history
        }
    }
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n--- Eğitim Tamamlandı ---")
    print(f"Klasör: {run_dir}")
    print(f"Model: {model_save_path}")
    print(f"Sonuçlar: results.json")

if __name__ == "__main__":
    main()

