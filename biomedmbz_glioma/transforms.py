import torch
import numpy as np

from monai import transforms

from .transforms_utils import ConvertToMultiChannelBasedOnBrats2023Classesd
from .transforms_utils import DownsampleLabelDeepSup
from .transforms_utils import LoadPreprocessedSampleNPY

def get_train_val_transforms(roi_size, aug_type=1):
    if aug_type == 1:
        train_transform = transforms.Compose(
            [
                LoadPreprocessedSampleNPY(keys=["image", "label", "meta"]),
                ConvertToMultiChannelBasedOnBrats2023Classesd(keys="label"),
                transforms.RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=[roi_size[0], roi_size[1], roi_size[2]],
                    random_size=False,
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
    elif aug_type == 2:
        # rand_elastic = transforms.Rand3DElasticd(
        #     keys=["image", "label"],
        #     mode=("bilinear", "nearest"),
        #     prob=0.50,
        #     sigma_range=(5, 8),
        #     magnitude_range=(50, 200),
        #     spatial_size=(128, 128, 128),
        #     # rotate_range=(np.pi / 36, np.pi / 36, np.pi),
        #     rotate_range=15 * np.pi/180,
        #     # scale_range=(0.15, 0.15, 0.15),
        #     scale_range=0.15,
        #     padding_mode="border",
        # )
        
        train_transform = transforms.Compose(
            [
                LoadPreprocessedSampleNPY(keys=["image", "label", "meta"]),
                ConvertToMultiChannelBasedOnBrats2023Classesd(keys="label"),
                transforms.RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=[roi_size[0], roi_size[1], roi_size[2]],
                    random_size=False,
                ),
                # rand_elastic,
                transforms.RandAffined(keys=["image", "label"], mode=("bilinear", "nearest"), prob=0.7,
                                       padding_mode="border", scale_range=0.15, rotate_range=15 * np.pi/180),
                transforms.RandScaleIntensityd(keys="image", factors=0.15, prob=0.7),
                transforms.RandAdjustContrastd(keys="image", gamma=(0.5, 2), prob=0.7),
            ]
        )
    elif aug_type == 3:
        train_transform = transforms.Compose(
            [
                LoadPreprocessedSampleNPY(keys=["image", "label", "meta"]),
                ConvertToMultiChannelBasedOnBrats2023Classesd(keys="label"),
                transforms.RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=[roi_size[0], roi_size[1], roi_size[2]],
                    random_size=False,
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                transforms.RandAffined(
                    keys=["image", "label"], prob=1.0,
                    mode=("bilinear", "nearest"), padding_mode="border",
                    scale_range=0.20, rotate_range=15 * np.pi/180, shear_range=0.20,
                ),
                transforms.RandScaleIntensityd(keys="image", factors=0.15, prob=1.0),
                transforms.RandAdjustContrastd(keys="image", gamma=(0.5, 1.5), prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )
    
    val_transform = transforms.Compose([
        LoadPreprocessedSampleNPY(keys=["image", "label", "meta"]),
        ConvertToMultiChannelBasedOnBrats2023Classesd(keys="label"),
    ])
    
    return train_transform, val_transform

def get_test_transform():
    test_transform = transforms.Compose(
        [
            LoadPreprocessedSampleNPY(keys=["image", "label", "meta"]),
        ]
    )
    
    return test_transform

def deep_sup_get_train_val_transforms(roi_size, deep_sup_levels=[]):
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBrats2023Classesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi_size[0], roi_size[1], roi_size[2]],
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi_size[0], roi_size[1], roi_size[2]],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            DownsampleLabelDeepSup(roi_size=roi_size, deep_sup_levels=deep_sup_levels),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBrats2023Classesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    
    return train_transform, val_transform