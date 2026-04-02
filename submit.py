import torch
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from model import get_model
from tqdm import tqdm

def generate_submission():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Cihaz: {device}")

    # Test CSV dosyasını oku
    test_csv_path = 'dataset/test.csv'
    test_img_dir = 'dataset/test'
    
    if not os.path.exists(test_csv_path):
        print(f"Hata: {test_csv_path} bulunamadı.")
        return
        
    df_test = pd.read_csv(test_csv_path)
    
    # Modeli yükle
    model = get_model(pretrained=False)
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("Model ağırlıkları (best_model.pth) başarıyla yüklendi!")
    except FileNotFoundError:
        print("Hata: 'best_model.pth' bulunamadı.")
        return
        
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    predictions = []
    
    print("Test verisi üzerinden tahminler (submission) oluşturuluyor...")
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        img_name = str(row['image_name']) + '.jpg'
        img_path = os.path.join(test_img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
                predictions.append(prob)
        except Exception as e:
            # Hata varsa (resim bozuk vb.) varsayılan 0.0 ekle
            predictions.append(0.0)
            
    # Submission Dataframe oluştur
    submission_df = pd.DataFrame({
        'image_name': df_test['image_name'],
        'target': predictions
    })
    
    submission_df.to_csv('submission.csv', index=False)
    print("\nHarika! Kaggle vb. yarışmalar için 'submission.csv' başarıyla oluşturuldu.")

if __name__ == "__main__":
    generate_submission()
