import torch
import torch.nn as nn
import torch.nn.functional as F

import monai
from monai.networks.blocks.convolutions import ResidualUnit
from monai.networks.layers.factories import Act


from .base import SegmentorWithTTA

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.downsampling_layer = nn.Sequential(
            nn.GroupNorm(in_channels // 8, in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
        )
        
        num_groups = in_channels / 8
        assert num_groups == int(num_groups)
        num_groups = int(num_groups)
        
        self.block = ResidualUnit(
            3,
            in_channels,
            in_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
            act=Act.GELU,
            norm=("GROUP", {"num_groups": num_groups, "affine": True}),
            dropout=0.0,
            bias=True,
            last_conv_only=False,
            adn_ordering="NDA",
        )
    
    def forward(self, x):
        x = self.block(x)
        return x, self.downsampling_layer(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        self.upsampling_layer = nn.Sequential(
            nn.GroupNorm(in_channels // 8, in_channels),
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2),
        )
        
        
        num_groups = out_channels / 8
        assert num_groups == int(num_groups)
        num_groups = int(num_groups)
        
        self.block = ResidualUnit(
            3,
            in_channels + skip_channels,
            out_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
            act=Act.GELU,
            norm=("GROUP", {"num_groups": num_groups, "affine": True}),
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
    def __init__(self, in_channels):
        super().__init__()
        
        num_groups = in_channels / 8
        assert num_groups == int(num_groups)
        num_groups = int(num_groups)
        
        self.block = ResidualUnit(
            3,
            in_channels,
            in_channels,
            strides=1,
            kernel_size=3,
            subunits=2,
            act=Act.GELU,
            norm=("GROUP", {"num_groups": num_groups, "affine": True}),
            dropout=0.0,
            bias=True,
            last_conv_only=False,
            adn_ordering="NDA",
        )
    
    def forward(self, x):
        return self.block(x)

class CNN_UNET(nn.Module):
    def __init__(self):
        super().__init__()
        
        list_channels = [8, 64, 128, 256, 320, 512]
        
        self.pre_conv = nn.Sequential(
            nn.InstanceNorm3d(4, affine=True),
            nn.Conv3d(4, list_channels[0], kernel_size=1),
        )
        
        self.encoders = nn.ModuleList([
            EncoderBlock(list_channels[i], list_channels[i+1]) \
            for i in range(len(list_channels) - 1)
        ])
        
        self.center = CenterBlock(list_channels[-1])
        
        self.decoders = nn.ModuleList([
            DecoderBlock(list_channels[-i], list_channels[-i-1], list_channels[-i-1]) \
            for i in range(1, len(list_channels))
        ])
        
        self.last_conv = nn.Sequential(
            # nn.GroupNorm(list_channels[0] // 8, list_channels[0]),
            # nn.InstanceNorm3d(, affine=True),
            nn.Conv3d(list_channels[0], 3, kernel_size=1),
        )
    
    def forward(self, x):
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
        
        return y