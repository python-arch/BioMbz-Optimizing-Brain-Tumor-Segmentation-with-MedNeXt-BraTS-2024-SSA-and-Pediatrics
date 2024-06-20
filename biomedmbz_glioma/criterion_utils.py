import torch.nn as nn

class DeepSupCriterion(nn.Module):
    def __init__(self, criterion, deep_sup_levels, deep_sup_weights):
        super().__init__()
        
        self.criterion = criterion
        self.deep_sup_levels  = deep_sup_levels
        self.deep_sup_weights = deep_sup_weights
    
    def forward(self, logits, target, deep_sup_logits, batch_data):
        loss = self.criterion(logits, target)
        for deep_sup_level, deep_sup_weight in \
            zip(self.deep_sup_levels, self.deep_sup_weights):
            deep_sup_target = batch_data[f'label_level_{deep_sup_level}']
            # print(deep_sup_logit.shape)
            loss += deep_sup_weight * self.criterion(deep_sup_logits[deep_sup_level], deep_sup_target)
        return loss