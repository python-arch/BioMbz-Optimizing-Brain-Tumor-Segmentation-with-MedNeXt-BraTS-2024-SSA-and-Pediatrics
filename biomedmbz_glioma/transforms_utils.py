import copy
import numpy as np
import torch

from collections.abc import Hashable, Mapping
from typing import Dict

from monai import transforms
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, Transform
from monai.utils.enums import TransformBackends

class LoadPreprocessedSampleNPY(MapTransform):
    def __call__(self, data):
        d = {
            "image": torch.from_numpy(np.load(data['image'])[:-1]),
            "meta" : torch.from_numpy(np.load(data['meta'])),
        }
        if 'label' in data:
            d['label'] = torch.from_numpy(np.load(data['label']))
        
        out = copy.deepcopy(data)
        for key, val in d.items():
            out[key] = val
        
        return out

# Reference:
# https://github.com/Project-MONAI/MONAI/blob/e4751964284959b4cd5086f626963ae9801189d3/monai/transforms/utility/array.py#L1078-L1098
class ConvertToMultiChannelBasedOnBrats2023Classes(Transform):
    """
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 3 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """
    
    
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    
    
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        
        
        result = [(img == 1) | (img == 3), (img == 1) | (img == 3) | (img == 2), img == 3]
        # merge labels 1 (tumor non-enh) and 3 (tumor enh) and 2 (large edema) to WT
        # label 3 is ET
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)

# Reference:
# https://github.com/Project-MONAI/MONAI/blob/e4751964284959b4cd5086f626963ae9801189d3/monai/transforms/utility/dictionary.py#L1342-L1363
class ConvertToMultiChannelBasedOnBrats2023Classesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`ConvertToMultiChannelBasedOnBrats2023Classes`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 3 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """
    
    
    backend = ConvertToMultiChannelBasedOnBrats2023Classes.backend
    
    
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBrats2023Classes()
    
    
    def __call__(self, data): # : Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

class DownsampleLabelDeepSup(MapTransform):
    def __init__(self, roi_size, deep_sup_levels=[]):
        assert min(deep_sup_levels) > 0
        
        self.deep_sup_levels = []
        for deep_sup_level in deep_sup_levels:
            key = f'label_level_{deep_sup_level}'
            spatial_size = [size // (2**deep_sup_level) for size in roi_size]
            transform = transforms.Resized(
                keys=key, spatial_size=spatial_size,
                mode='nearest',
            )
            self.deep_sup_levels.append((key, transform))
    
    def __call__(self, inputs):
        if len(self.deep_sup_levels) < 1:
            return inputs
        
        for key, transform in self.deep_sup_levels:
            inputs[key] = copy.deepcopy(inputs['label'])
            inputs = transform(inputs)
        
        return inputs