import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLossMultiLabel(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification (Ridnik et al., ICCV 2021).
    Optimized for severely imbalanced multi-label problems.
    
    - gamma_pos: Focusing parameter for positive samples (default: 0, preserves positive signal).
    - gamma_neg: Focusing parameter for negative samples (default: 3.0, heavily discounts easy negatives).
    - clip: Margin for shifting negative probabilities (default: 0.05). If p < clip, gradient = 0.
    - eps: Epsilon for numerical stability.
    - pos_weight: Optional tensor of shape (num_classes,) to scale positive class contributions.
    """
    def __init__(self, gamma_neg=3.0, gamma_pos=0.0, clip=0.05, eps=1e-8, pos_weight=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        logits: (batch_size, num_classes) - raw model output (pre-sigmoid)
        targets: (batch_size, num_classes) - binary 0 or 1 float targets
        """
        targets = targets.float()
        probs = torch.sigmoid(logits)
        
        # Positive Loss: - targets * (1 - p)^gamma_pos * log(p + eps)
        probs_pos = probs.clamp(min=self.eps, max=1.0 - self.eps)
        loss_pos = targets * torch.log(probs_pos)
        if self.gamma_pos > 0:
            loss_pos = loss_pos * ((1.0 - probs_pos) ** self.gamma_pos)
            
        # Optional per-class positive weighting
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)
            loss_pos = loss_pos * pw

        # Negative Loss: - (1 - targets) * (p_m)^gamma_neg * log(1 - p_m + eps)
        # Shift probabilities by margin clip
        if self.clip > 0:
            probs_neg = (probs - self.clip).clamp(min=0.0)
        else:
            probs_neg = probs
            
        loss_neg = (1.0 - targets) * torch.log((1.0 - probs_neg).clamp(min=self.eps))
        if self.gamma_neg > 0:
            loss_neg = loss_neg * (probs_neg ** self.gamma_neg)

        loss = - (loss_pos + loss_neg)
        return loss.mean()
