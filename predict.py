import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import sys

def predict_image(image_path, model_path="best_model.pth"):
    # Cihaz ayarı
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Sınıf İsimleri
    classes = {0: "İyi Huylu (Benign)", 1: "Kanserli (Malignant)"}

    # Modeli Yükle
    print("Model hazırlanıyor...")
    model = get_model(pretrained=False) # Eğitim aşamasında değilsek pretrained indirmemize gerek yok
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Hata: '{model_path}' bulunamadı. Lütfen önce eğitimi tamamlayın veya modeli kaydedin.")
        return
        
    model = model.to(device)
    model.eval()

    # Ön İşleme (Validation ile aynı olmalı)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Görüntüyü Oku
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Görüntü okunurken hata oluştu: {e}")
        return

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Tahmin (Inference)
    print(f"'{image_path}' tahmin ediliyor...")
    with torch.no_grad():
        output = model(input_tensor)
        # Çıktıyı 0-1 arası olasılığa dönüştür (Sigmoid)
        prob = torch.sigmoid(output).item()
        
    # Yorumlama (Threshold: 0.5)
    prediction = 1 if prob >= 0.5 else 0
    
    print("-" * 30)
    print(f"Tahmin: {classes[prediction]}")
    print(f"Kanser Olasılığı: %{prob * 100:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python predict.py <resim_yolu>")
        print("Örnek: python predict.py dataset/test/ornek1.jpg")
    else:
        predict_image(sys.argv[1])
