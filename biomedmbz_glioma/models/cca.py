import torch
import torch.nn as nn

class CrossChannelAttn(nn.Module):
    def __init__(
        self,
        num_channels,
        kernel_size=3,
        skip_con=True,
    ):
        super().__init__()
        
        self.skip_con = skip_con
        
        padding = (kernel_size - 1) / 2
        assert padding == int(padding)
        padding = int(padding)
        
        self.conv = nn.Conv3d(
            2*num_channels,
            2*num_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=2,
            bias=False,
        )
        
        self.gn_in  = nn.GroupNorm(2*8, 2*num_channels)
        self.gn_out = nn.GroupNorm(2*8, 2*num_channels)
        
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def forward(self, up, skip):
        x = torch.cat((up, skip), 1)
        
        x = self.gn_in(x)
        x = self.gelu(x)
        x = self.conv(x)
        x = self.gn_out(x)
        
        _, C, _, _, _ = x.shape
        
        x_up   = x[:, :C//2]
        x_skip = x[:, C//2:]
        
        x = x_up * x_skip
        x = self.avgpool(x)
        x = self.sigmoid(x)
        
        x = skip * x
        
        if self.skip_con == True:
            x = x + skip
        
        return x
