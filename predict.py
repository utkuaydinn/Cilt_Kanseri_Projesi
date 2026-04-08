import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import sys

def predict_image(image_path, model_path="best_model.pth"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    classes = {0: "Iyi Huylu (Benign)", 1: "Kanserli (Malignant)"}

    print("Model hazirlaniyor...")
    model = get_model(pretrained=False)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Hata: '{model_path}' bulunamadi.")
        return
        
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Goruntu okunurken hata: {e}")
        return

    input_tensor = transform(image).unsqueeze(0).to(device)

    print(f"'{image_path}' tahmin ediliyor...")
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        
    prediction = 1 if prob >= 0.5 else 0
    
    print("-" * 30)
    print(f"Tahmin: {classes[prediction]}")
    print(f"Kanser Olasiligi: %{prob * 100:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python predict.py <resim_yolu>")
        print("Ornek: python predict.py dataset/test/ornek1.jpg")
    else:
        predict_image(sys.argv[1])
