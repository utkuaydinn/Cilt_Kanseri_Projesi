import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders
from model import get_model
from trainer import train_model, plot_history

def main():
    #MPS (Metal Performance Shaders) Cihaz Ayarı
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    # Hiperparametreler
    csv_path = 'dataset/train.csv'
    img_dir = 'dataset/train'
    batch_size = 32
    img_size = 224
    num_epochs = 5
    learning_rate = 1e-4

    # 1. Dataloader'ları Yükle
    print("Veri yükleniyor ve Data Augmentation uygulanıyor...")
    train_loader, val_loader, train_df, val_df = get_dataloaders(
        csv_path=csv_path, img_dir=img_dir, batch_size=batch_size, img_size=img_size
    )
    print(f"Eğitim Seti: {len(train_df)} örnek | Doğrulama Seti: {len(val_df)} örnek")

    # 2. Modeli Yükle
    print("EfficientNet-V2-S (Google) Modeli Yükleniyor...")
    model = get_model(pretrained=True)

    # 3. Kayıp Fonksiyonu (Focal Loss)
    # Sınıf dengesizliğinde ezbere giden 55 kat BCE cezası yerine zeka odaklı Focal Loss
    from utils import FocalLoss
    print("Focal Loss (Alpha: 0.85, Gamma: 2.0) kullanılıyor...")
    criterion = FocalLoss(alpha=0.85, gamma=2.0)

    # Optimizasyon Algoritması (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 4. Eğitimi Başlat
    print("Eğitim Başlıyor...")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs
    )

    # Eğitilen modeli disk'e kaydet (Daha sonra tahminleme yapmak için)
    save_path = "best_model.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model başarıyla {save_path} olarak kaydedildi!")

    # Eğitim geçmişini grafiğe çiz
    plot_history(history)

if __name__ == "__main__":
    main()
