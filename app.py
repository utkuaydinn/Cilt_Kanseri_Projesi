import gradio as gr
import torch
from torchvision import transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2

from model import get_model

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False
    print("Uyari: pytorch-grad-cam yuklu degil. Isı haritasi olusturulamayacak.")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_inference_model():
    # Kullanicinin tercihine gore daha az kaynak tuketen EfficientNet-V2-S yukleniyor
    model = get_model(model_name="efficientnet", pretrained=False)
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("Model basariyla yuklendi (best_model.pth)")
    except FileNotFoundError:
        print("UYARI: 'best_model.pth' bulunamadi. Model egitimsiz durumdadir.")
    
    model = model.to(device)
    model.eval()
    return model

model = load_inference_model()

# EfficientNet icin son conv katmani genellikle features icindeki son blocktur
# ResNet icin layer4'tur
if hasattr(model, 'features'):
    target_layers = [model.features[-1]]
elif hasattr(model, 'layer4'):
    target_layers = [model.layer4[-1]]
else:
    target_layers = []

if HAS_GRAD_CAM and target_layers:
    cam = GradCAM(model=model, target_layers=target_layers)
else:
    cam = None

def predict(image):
    if image is None:
        return "Lutfen bir gorsel yukleyin.", None
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    pil_image = Image.fromarray(image).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        
    prediction_text = "Kanserli (Malignant)" if prob >= 0.5 else "Iyi Huylu (Benign)"
    confidence = prob if prob >= 0.5 else 1 - prob
    result_text = f"Tahmin: {prediction_text}\nGuven Orani: %{confidence*100:.2f}"
    
    cam_image = None
    if cam is not None:
        try:
            # Sadece 1 cikti oldugu icin hedefleri None birakiyoruz
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            rgb_img = cv2.resize(image, (224, 224))
            rgb_img = np.float32(rgb_img) / 255
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        except Exception as e:
            print(f"Grad-CAM hatasi: {e}")
            cam_image = image
    else:
        cam_image = image
        
    return result_text, cam_image

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Cilt Lezyonu Gorseli Yukleyin"),
    outputs=[
        gr.Textbox(label="Tahmin Sonucu", lines=2),
        gr.Image(label="Modelin Baktigi Alan (Grad-CAM Isı Haritası)")
    ],
    title="Cilt Kanseri Tespiti - AI Arayuzu",
    description="Bu uygulama, bir cilt lezyonunun kanserli olup olmadigini tahmin eder ve modelin karar verirken resmin neresine odaklandigini isi haritasi ile gosterir."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
