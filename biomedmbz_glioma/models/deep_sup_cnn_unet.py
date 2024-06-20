import torch
import torch.nn as nn
import torch.nn.functional as F

import monai
from monai.networks.blocks.convolutions import ResidualUnit
from monai.networks.layers.factories import Act, Norm


from .base import SegmentorWithTTA

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeat=1):
        super().__init__()
        
        self.downsampling_layer = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
        )
        
        self.block = nn.Sequential(*[
            ResidualUnit(
            3,
            in_channels,
            in_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
            act=Act.GELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            last_conv_only=False,
            adn_ordering="NDA",
        ) for _ in range(repeat)
        ])
    
    def forward(self, x):
        x = self.block(x)
        return x, self.downsampling_layer(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        self.upsampling_layer = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2),
        )
        
        self.block = ResidualUnit(
            3,
            in_channels + skip_channels,
            out_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
            act=Act.GELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            last_conv_only=False,
            adn_ordering="NDA",
        )
    
    def forward(self, x, skip):
        x = self.upsampling_layer(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        
        return x

class CenterBlock(nn.Module):
    def __init__(self, in_channels, repeat=1):
        super().__init__()
        
        self.block = nn.Sequential(*[
            ResidualUnit(
            3,
            in_channels,
            in_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
            act=Act.GELU,
            norm=Norm.INSTANCE,
            dropout=0.0,
            bias=True,
            last_conv_only=False,
            adn_ordering="NDA",
        ) for _ in range(repeat)
        ])
    
    def forward(self, x):
        return self.block(x)

class DeepSupCNNUNet(nn.Module):
    def __init__(self, deep_sup_levels=(1, 2, 3)):
        super().__init__()
        
        list_enc_channels = [32, 64, 128, 256, 320, 512]
        list_enc_repeat = [1, 1, 1, 2, 4, 2]
        
        list_dec_channels = [16, 32, 64, 128, 256, 512]
        
        self.pre_conv = nn.Sequential(
            nn.InstanceNorm3d(4),
            nn.Conv3d(4, list_enc_channels[0], kernel_size=1),
        )
        
        self.encoders = nn.ModuleList([
            EncoderBlock(list_enc_channels[i], list_enc_channels[i+1], repeat=repeat) \
            for i, repeat in enumerate(list_enc_repeat[:-1])
        ])
        
        self.center = CenterBlock(list_enc_channels[-1], repeat=list_enc_repeat[-1])
        
        self.decoders = nn.ModuleList([
            DecoderBlock(list_dec_channels[-i], list_enc_channels[-i-1], list_dec_channels[-i-1]) \
            for i in range(1, len(list_dec_channels))
        ])
        
        self.last_conv = nn.Sequential(
            nn.Conv3d(list_dec_channels[0], 3, kernel_size=1),
        )
        
        assert min(deep_sup_levels) > 0
        assert max(deep_sup_levels) < len(list_dec_channels)
        
        self.deep_sup_levels = deep_sup_levels
        # print([list_dec_channels[deep_sup_level] for deep_sup_level in self.deep_sup_levels])
        # breakpoint()
        self.aux_conv_layer = nn.ModuleDict({
            str(deep_sup_level): nn.Conv3d(list_dec_channels[deep_sup_level], 3, kernel_size=1)
            for deep_sup_level in self.deep_sup_levels
        })
    
    def forward(self, x):
        return self.forward_features(x)[0]
    
    def forward_features(self, x):
        x = self.pre_conv(x)
        
        skip_features = []
        for encoder in self.encoders:
            f, x = encoder(x)
            skip_features.append(f)
        
        x = self.center(x)
        
        out_features = []
        for decoder, skip_f in zip(self.decoders, skip_features[::-1]):
            x = decoder(x, skip_f)
            out_features.append(x)
        
        y = self.last_conv(out_features[-1])
        
        out_features = out_features[::-1]
        # for deep_sup_level in self.deep_sup_levels:
        #     print(out_features[deep_sup_level].shape)
        #     print(self.aux_conv_layer[str(deep_sup_level)])
        out_features = {
            deep_sup_level: self.aux_conv_layer[str(deep_sup_level)](out_features[deep_sup_level])
            for deep_sup_level in self.deep_sup_levels
        }
        
        return y, out_features