import torch
import torch.optim as optim
import os
import argparse
import json
from datetime import datetime
from dataset import get_dataloaders
from model import get_model
from trainer import train_model, plot_history
from utils import FocalLoss

def main():
    parser = argparse.ArgumentParser(description='Cilt Kanseri Tespiti Egitim Scripti')
    parser.add_argument('--model', type=str, default='efficientnet', choices=['efficientnet', 'resnet'], 
                        help='Egitilecek model mimarisi')
    parser.add_argument('--batch', type=int, default=32, help='Batch size degeri')
    parser.add_argument('--epochs', type=int, default=5, help='Epoch sayisi')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_argument_list if hasattr(parser, 'parse_argument_list') else parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Sistem: NVIDIA GPU (CUDA) algilandi. Cihaz: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Sistem: Apple Silicon (MPS) algilandi.")
    else:
        device = torch.device("cpu")
        print("Sistem: GPU bulunamadi, CPU kullaniliyor.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_B{args.batch}_{timestamp}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Bu egitim '{run_dir}' klasorune kaydedilecek.")

    csv_path = 'dataset/train.csv'
    img_dir = 'dataset/train'
    img_size = 224

    print(f"Veri yukleniyor (Batch: {args.batch}, Size: {img_size})...")
    train_loader, val_loader, train_df, val_df = get_dataloaders(
        csv_path=csv_path, img_dir=img_dir, batch_size=args.batch, img_size=img_size
    )

    model = get_model(model_name=args.model, pretrained=True)

    criterion = FocalLoss(alpha=0.85, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Egitim Basliyor: {args.model} | Epochs: {args.epochs}")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs
    )

    model_save_path = os.path.join(run_dir, "best_model.pth")
    torch.save(trained_model.state_dict(), model_save_path)
    
    plot_save_path = os.path.join(run_dir, "training_history.png")
    plot_history(history, save_path=plot_save_path)
    
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

    print(f"\n--- Egitim Tamamlandi ---")
    print(f"Klasor: {run_dir}")
    print(f"Model: {model_save_path}")
    print(f"Sonuclar: results.json")

if __name__ == "__main__":
    main()
