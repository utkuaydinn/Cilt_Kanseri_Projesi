import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name="efficientnet", pretrained=True):
    if model_name.lower() == "efficientnet":
        print(f"Sistem: EfficientNet-V2-S yukleniyor (Pretrained: {pretrained})...")
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 1)
        
    elif model_name.lower() == "resnet":
        print(f"Sistem: ResNet-50 yukleniyor (Pretrained: {pretrained})...")
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        
    else:
        raise ValueError(f"Hata: '{model_name}' desteklenmiyor. Secenekler: 'efficientnet', 'resnet'")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Hazir! Egitilebilir Parametre Sayisi: {trainable_params:,}")
    
    return model
