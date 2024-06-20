import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional

from monai.networks.nets import SegResNet

from .cca import CrossChannelAttn
from .mwcnn import ModalityWiseCNN

class SegResNetC(SegResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{kwarg: kwargs[kwarg] for kwarg in kwargs if (kwarg != "apply_sigmoid") and (kwarg != "apply_cca") and (kwarg != "cca_skip_con") and (kwarg != "apply_mwcnn") and (kwarg != 'deep_sup')})
        
        if (kwargs['apply_cca'] == True) and (kwargs['apply_mwcnn'] == True):
            raise ValueError
        
        self.apply_sigmoid = kwargs['apply_sigmoid']
        
        self.apply_cca = kwargs['apply_cca']
        if self.apply_cca == True:
            cca_layers = []
            for i in range(len(kwargs['blocks_up'])):
                cca_layers.append(CrossChannelAttn(2**i * kwargs['init_filters'], skip_con=kwargs['cca_skip_con']))
            self.cca_layers = nn.ModuleList(cca_layers[::-1])
        
        self.apply_mwcnn = kwargs['apply_mwcnn']
        if self.apply_mwcnn == True:
            n_modality = 4
            list_out_channels = [2**i * kwargs['init_filters'] for i in range(len(kwargs['blocks_down']))]
            list_repeat = kwargs['blocks_down']
            list_channels = [2*c for c in list_out_channels]
            
            self.mwcnn = ModalityWiseCNN(n_modality, list_channels, list_repeat, list_out_channels)
        
        if 'deep_sup' in kwargs:
            self.deep_sup = kwargs['deep_sup']
            conv_out_ds = []
            for i in range(1, len(kwargs['blocks_down'])):
                conv_out_ds.append(nn.Conv3d(2**i * kwargs['init_filters'], kwargs['out_channels'], kernel_size=1))
            self.conv_out_ds = nn.ModuleList(conv_out_ds[::-1])
        else:
            self.deep_sup = False
    
    def _forward(self, x):
        if self.apply_mwcnn == True:
            aux_x, aux = self.mwcnn(x)
            aux.reverse()
            
            x, down_x = self.encode(x)
            down_x.reverse()
            
            x = x + aux_x
            
            x = self.decode(x, down_x, aux)
            
            return x
        else:
            return super().forward(x)
    
    def forward(self, x):
        logits = self._forward(x)
        if self.apply_sigmoid == False:
            return logits
        else:
            return F.sigmoid(logits)
    
    def decode(self, x: torch.Tensor, down_x: List[torch.Tensor], aux: Optional[torch.Tensor]=None) -> torch.Tensor:
        if self.apply_cca == True:
            for i, (up, upl, ccal) in enumerate(zip(self.up_samples, self.up_layers, self.cca_layers)):
                x = up(x)
                x = x + ccal(x, down_x[i + 1])
                x = upl(x)
            
            if self.use_conv_final:
                x = self.conv_final(x)
            
            return x
        elif self.apply_mwcnn == True:
            assert aux is not None
            
            for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
                x = up(x) + down_x[i + 1] + aux[i + 1]
                x = upl(x)
            
            if self.use_conv_final:
                x = self.conv_final(x)
            
            return x
        elif (self.deep_sup==True) and (self.training==True):
            out = []
            for i, (up, upl, oconv) in enumerate(zip(self.up_samples, self.up_layers, self.conv_out_ds)):
                out.append(oconv(x))
                x = up(x) + down_x[i + 1]
                x = upl(x)
            if self.use_conv_final:
                x = self.conv_final(x)
            out.append(x)
            return out[::-1]
        else:
            return super().decode(x, down_x)