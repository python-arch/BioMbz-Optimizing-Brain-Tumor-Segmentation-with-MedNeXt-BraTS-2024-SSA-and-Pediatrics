import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import (
    DiceLoss, DiceFocalLoss,
)

from monai.data.meta_tensor import MetaTensor

class CriterionWrapper(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        
        self.criterion = criterion
    
    def forward(self, logits, target):
        if type(logits) in [torch.Tensor, MetaTensor]:
            return self.criterion(logits, target)
        elif type(logits) == list:
            if logits[0].shape[-1] < logits[1].shape[-1]:
                logits = logits[::-1]
            
            weights = [1 / (2 ** i) for i in range(len(logits))]
            
            loss = weights[0] * self.criterion(logits[0], target)
            for weight, _logits in zip(weights[1:], logits[1:]):
                loss += weight * self.criterion(_logits, self.interpolate_label(target, scale=weight))
            
            return loss
        else:
            raise ValueError(f'type(logits) {type(logits)} is not recognized')
    
    def interpolate_label(self, label, scale):
        dtype = label.dtype
        
        size = [round(s * scale) for s in label.shape[2:]]
        
        out = F.interpolate(label.float(), size)
        out = out.type(dtype)
        
        return out

def get_loss_fn(loss_type, mean_batch):
    if loss_type == 1:
        loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True, batch=mean_batch)
    elif loss_type == 2:
        loss_fn = DiceFocalLoss(to_onehot_y=False, sigmoid=True, batch=mean_batch, gamma=0.0)
    elif loss_type == 3:
        loss_fn = DiceFocalLoss(to_onehot_y=False, sigmoid=True, batch=mean_batch, gamma=2.0)
    else:
        raise ValueError
    
    return loss_fn