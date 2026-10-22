import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCEWithLogitsLoss içindeki gibi sigmoid alır
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Olasılıkları hesapla (pt)
        pt = torch.exp(-BCE_loss)
        
        # Focal Loss formülü: alpha * (1 - pt)^gamma * BCE
        # Alpha uygulanırken pozitif class için alpha, negatif class için (1-alpha)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        F_loss = alpha_t * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
