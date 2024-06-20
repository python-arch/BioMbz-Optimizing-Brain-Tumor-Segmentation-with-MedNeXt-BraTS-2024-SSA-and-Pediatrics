import torch.nn as nn

from monai.transforms import Activations

from nnunet_mednext import create_mednext_v1

def get_model_mednext_b3(*args, **kwargs):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.model = create_mednext_v1(
                num_input_channels=4,
                num_classes=3,
                model_id='B',
                kernel_size=3,
                deep_supervision=True,
                checkpoint_style=None,
            )
            self.apply_sigmoid = True
            
            self.fn_sigmoid = Activations(sigmoid=True)
        
        def forward(self, x):
            preds = self.model.forward(x)
            if self.apply_sigmoid == True:
                preds = self.fn_sigmoid(preds)
            return preds
    
    return Model()

def get_model_mednext_m3(*args, **kwargs):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.model = create_mednext_v1(
                num_input_channels=4,
                num_classes=3,
                model_id='M',
                kernel_size=3,
                deep_supervision=True,
                checkpoint_style=None,
            )
            self.apply_sigmoid = True
            
            self.fn_sigmoid = Activations(sigmoid=True)
        
        def forward(self, x):
            preds = self.model.forward(x)
            if self.apply_sigmoid == True:
                preds = self.fn_sigmoid(preds)
            return preds
    
    return Model()

def get_model_mednext_m5(*args, **kwargs):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.model = create_mednext_v1(
                num_input_channels=4,
                num_classes=3,
                model_id='M',
                kernel_size=5,
                deep_supervision=True,
                checkpoint_style=None,
            )
            self.apply_sigmoid = True
            
            self.fn_sigmoid = Activations(sigmoid=True)
        
        def forward(self, x):
            preds = self.model.forward(x)
            if self.apply_sigmoid == True:
                preds = self.fn_sigmoid(preds)
            return preds
    
    return Model()

def get_model_mednext_b5(*args, **kwargs):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.model = create_mednext_v1(
                num_input_channels=4,
                num_classes=3,
                model_id='B',
                kernel_size=5,
                deep_supervision=True,
                checkpoint_style=None,
            )
            self.apply_sigmoid = True
            
            self.fn_sigmoid = Activations(sigmoid=True)
        
        def forward(self, x):
            preds = self.model.forward(x)
            if self.apply_sigmoid == True:
                preds = self.fn_sigmoid(preds)
            return preds
    
    return Model()

