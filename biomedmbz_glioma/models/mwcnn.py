import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, n_groups, channels, bottleneck_scale=4):
        super().__init__()
        
        int_channels = channels // bottleneck_scale
        
        self.gn1   = nn.GroupNorm(n_groups*4, channels)
        self.conv1 = nn.Conv3d(channels, int_channels, 1, groups=n_groups)
        
        self.gn2   = nn.GroupNorm(n_groups*4, int_channels)
        self.conv2 = nn.Conv3d(int_channels, int_channels, 3, 1, 1, groups=n_groups)
        
        self.gn3   = nn.GroupNorm(n_groups*4, int_channels)
        self.conv3 = nn.Conv3d(int_channels, channels, 1, groups=n_groups)
    
    def forward(self, x):
        identity = x
        
        x = self.conv1(F.relu(self.gn1(x), inplace=True))
        x = self.conv2(F.relu(self.gn2(x), inplace=True))
        x = self.conv3(F.relu(self.gn3(x), inplace=True))
        
        x += identity
        
        return x

class DownsamplingBlock(nn.Module):
    def __init__(self, n_groups, in_channels, out_channels):
        super().__init__()
        
        self.gn   = nn.GroupNorm(n_groups*4, in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, groups=n_groups)
    
    def forward(self, x):
        x = self.gn(x)
        x = self.conv(x)
        
        return x

class SkipConBlock(nn.Module):
    def __init__(self, n_groups, in_channels, out_channels):
        super().__init__()
        
        self.gn   = nn.GroupNorm(n_groups*4, in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, 1, groups=n_groups)
    
    def forward(self, x):
        x = self.gn(x)
        x = self.conv(x)
        
        return x

class SingleModalityCNN(nn.Module):
    def __init__(self, list_channels, list_repeat, list_out_channels):
        super().__init__()
        
        assert len(list_channels) == len(list_out_channels)
        assert len(list_channels) == len(list_repeat)
        
        n_modality = 1
        
        self.pre_conv = nn.Sequential(
            nn.InstanceNorm3d(n_modality, affine=True),
            nn.Conv3d(n_modality, list_channels[0], kernel_size=1, groups=n_modality),
        )
        
        res_blocks = []
        for c, r in zip(list_channels, list_repeat):
            res_blocks.append(nn.Sequential(*[
                ResBlock(n_modality, c, bottleneck_scale=4)
                for _ in range(r)
            ]))
        self.res_blocks = nn.ModuleList(res_blocks)
        
        skip_blocks = []
        for ci, co in zip(list_channels, list_out_channels):
            skip_blocks.append(SkipConBlock(n_modality, ci, co))
        self.skip_blocks = nn.ModuleList(skip_blocks)
        
        down_blocks = []
        for i in range(len(list_channels)-1):
            down_blocks.append(DownsamplingBlock(n_modality, list_channels[i], list_channels[i+1]))
        self.down_blocks = nn.ModuleList(down_blocks)
    
    def forward(self, x):
        x = self.pre_conv(x)
        
        skip_features = []
        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x)
            x_int = self.skip_blocks[i](x)
            skip_features.append(x_int)
            if i < len(self.res_blocks) - 1:
                x = self.down_blocks[i](x)
        
        return skip_features

class ModalityWiseCNN(nn.Module):
    def __init__(self, n_modality, list_channels, list_repeat, list_out_channels):
        super().__init__()
        
        self.n_modality = n_modality
        
        s_list_channels = [c//n_modality for c in list_channels]
        
        self.blocks = nn.ModuleList([SingleModalityCNN(s_list_channels, list_repeat, list_out_channels)
                                     for _ in range(n_modality)])
    
    def forward(self, x):
        skip_features = self.blocks[0](x[:, 0:1])
        for i in range(1, self.n_modality):
            tmp = self.blocks[i](x[:, i:(i+1)])
            for j in range(len(skip_features)):
                skip_features[j] += tmp[j]
        return skip_features[-1], skip_features

# class ModalityWiseCNN(nn.Module):
#     def __init__(self, n_modality, list_channels, list_repeat, list_out_channels):
#         super().__init__()
        
#         assert len(list_channels) == len(list_out_channels)
#         assert len(list_channels) == len(list_repeat)
        
#         self.pre_conv = nn.Sequential(
#             nn.InstanceNorm3d(n_modality, affine=True),
#             nn.Conv3d(n_modality, list_channels[0], kernel_size=1, groups=n_modality),
#         )
        
#         res_blocks = []
#         for c, r in zip(list_channels, list_repeat):
#             res_blocks.append(nn.Sequential(*[
#                 ResBlock(n_modality, c, bottleneck_scale=4)
#                 for _ in range(r)
#             ]))
#         self.res_blocks = nn.ModuleList(res_blocks)
        
#         skip_blocks = []
#         for ci, co in zip(list_channels, list_out_channels):
#             skip_blocks.append(SkipConBlock(n_modality, ci, co))
#         self.skip_blocks = nn.ModuleList(skip_blocks)
        
#         down_blocks = []
#         for i in range(len(list_channels)-1):
#             down_blocks.append(DownsamplingBlock(n_modality, list_channels[i], list_channels[i+1]))
#         self.down_blocks = nn.ModuleList(down_blocks)
    
#     def forward(self, x):
#         x = self.pre_conv(x)
        
#         skip_features = []
#         for i in range(len(self.res_blocks)):
#             x = self.res_blocks[i](x)
#             x_int = self.skip_blocks[i](x)
#             skip_features.append(x_int)
#             if i < len(self.res_blocks) - 1:
#                 x = self.down_blocks[i](x)
        
#         return x_int, skip_features