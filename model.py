import torch
import torch.nn as nn
from torchvision import models

def get_model(pretrained=True):
    # Google'ın devrim niteliğindeki yeni nesil EfficientNet V2 S modelini çekiyoruz
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    
    # EfficientNet sınıflandırıcı yapısı: classifier dizisidir, 1. index asıl lineer katmandır.
    num_ftrs = model.classifier[1].in_features
    # Binary Classification (BCEWithLogitsLoss / FocalLoss) için sadece '1' çıkışa (Kanser Mi?) ihtiyacımız var
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    
    return model
