import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentorWithTTA(nn.Module):
    def __init__(self, model):
        self.model = model
    
    def forward(self, x):
        if self.training:
            return self.model.forward(x)
        else:
            return self.segment_with_tta(x)
    
    def segment_with_flip(self, x, dims):
        y = self.model.forward(torch.flip(x, dims))
        y = torch.flip(y, dims)
        y = F.sigmoid(y)
        
        return y
    
    def segment_with_tta(self, x):
        flip_combs = [
            [], [2], [3], [4],
            [2,3], [2,4], [3,4], [2,3,4],
        ]
        
        y = None
        
        for dims in flip_combs:
            if not torch.is_tensor(y):
                y = self.segment_with_flip(x, dims)
            else:
                y += self.segment_with_flip(x, dims)
        
        y = y / len(flip_combs)
        
        return y