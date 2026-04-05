import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name="efficientnet", pretrained=True):
    """
    Model seçici: EfficientNet-V2-S veya ResNet-50 yükler.
    Kullanım: model = get_model(model_name="resnet", pretrained=True)
    """
    if model_name.lower() == "efficientnet":
        print(f"Sistem: EfficientNet-V2-S yükleniyor (Pretrained: {pretrained})...")
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        
        # Classifier kısmını ikili sınıflandırma (Binary) için güncelliyoruz
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 1)
        
    elif model_name.lower() == "resnet":
        print(f"Sistem: ResNet-50 yükleniyor (Pretrained: {pretrained})...")
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        
        # ResNet'te son katman 'fc' olarak adlandırılır
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        
    else:
        raise ValueError(f"Hata: '{model_name}' desteklenmiyor. Seçenekler: 'efficientnet', 'resnet'")
    
    # Modelle ilgili kısa bilgi
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Hazır! Eğitilebilir Parametre Sayısı: {trainable_params:,}")
    
    return model

