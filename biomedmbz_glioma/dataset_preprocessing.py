import os
import math
import numpy as np
import nibabel
import shutil
import tqdm

from monai import transforms
from multiprocessing import Pool
from pathlib import Path

def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)

def load_modalities_and_merge(directory, example_id, list_modalities=["t2f", "t1n", "t1c", "t2w"]):
    modalities = [
        nibabel.load(os.path.join(directory, example_id, f'{example_id}-{modality}.nii.gz'))
        for modality in list_modalities
    ]
    affine, header = modalities[0].affine, modalities[0].header
    
    vol = np.stack([get_data(modality, "int16") for modality in modalities], axis=-1)
    vol = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
    
    return vol

def load_seg_label(directory, example_id):
    seg = nibabel.load(os.path.join(directory, example_id, f"{example_id}-seg.nii.gz"))
    affine, header = seg.affine, seg.header
    
    seg = get_data(seg, "unit8")
    seg = nibabel.nifti1.Nifti1Image(seg, affine, header=header)
    
    return seg

def crop_foreground(image, label=None):
    orig_shape = image.shape[1:]
    bbox = transforms.utils.generate_spatial_bounding_box(image)
    image = transforms.SpatialCrop(roi_start=bbox[0], roi_end=bbox[1])(image)
    image_metadata = np.vstack([bbox, orig_shape, image.shape[1:]])
    label = transforms.SpatialCrop(roi_start=bbox[0], roi_end=bbox[1])(label) \
        if label is not None else None
    
    return image, label, image_metadata

def normalize_intensity(image):
    fn = transforms.NormalizeIntensity(nonzero=True, channel_wise=True)
    return fn(image)

def calculate_pad_shape(image, patch_size):
    assert patch_size == [128, 128, 128]
    
    min_shape = patch_size
    image_shape = image.shape[1:]
    pad_shape = [max(mshape, ishape) for mshape, ishape in zip(min_shape, image_shape)]
    
    return pad_shape

def pad(image, padding):
    pad_d, pad_w, pad_h = padding
    return np.pad(
        image,
        (
            (0, 0),
            (math.floor(pad_d), math.ceil(pad_d)),
            (math.floor(pad_w), math.ceil(pad_w)),
            (math.floor(pad_h), math.ceil(pad_h)),
        ),
    )

def standardize(image, label, patch_size):
    pad_shape = calculate_pad_shape(image, patch_size)
    image_shape = image.shape[1:]
    if pad_shape != image_shape:
        paddings = [(pad_sh - image_sh) / 2 for (pad_sh, image_sh) in zip(pad_shape, image_shape)]
        image = pad(image, paddings)
        label = pad(label, paddings)
    return image, label

def encode_foregrounds(image):
    mask = np.zeros(image.shape[1:], dtype=np.float32)
    for i in range(image.shape[0]):
        ones = np.where(image[i] > 0)
        mask[ones] += 1.0
    mask[mask > 0] = 1.0
    image = normalize_intensity(image).astype(np.float32)
    mask = np.expand_dims(mask, 0)
    image = np.concatenate([image, mask])
    return image

def preprocess_sample(directory, example_id, patch_size=[128, 128, 128], list_modalities=["t2f", "t1n", "t1c", "t2w"]):
    vol = load_modalities_and_merge(directory, example_id, list_modalities)
    
    image = vol.get_fdata().astype(np.float32)
    image = np.transpose(image, (3, 0, 1, 2))
    image_spacing = vol.header["pixdim"][1:4].tolist()
    
    if os.path.exists(os.path.join(directory, example_id, f"{example_id}-seg.nii.gz")):
        label = load_seg_label(directory, example_id).get_fdata().astype(np.uint8)
        label = np.expand_dims(label, 0)
    else:
        label = None
    
    assert image_spacing == [1.0, 1.0, 1.0]
    
    image, label, image_metadata = crop_foreground(image, label)
    
    image = normalize_intensity(image)
    
    if label is not None:
        image, label = standardize(image, label, patch_size)
    
    image = encode_foregrounds(image)
    
    return image, label, image_metadata

def preprocessing_and_save(target_directory, source_directory, example_id, patch_size=[128, 128, 128], list_modalities=["t2f", "t1n", "t1c", "t2w"]):
    image, label, image_metadata = preprocess_sample(source_directory, example_id, patch_size, list_modalities)
    
    np.save(os.path.join(target_directory, f"{example_id}_x.npy"), image, allow_pickle=False)
    np.save(os.path.join(target_directory, f"{example_id}_meta.npy"), image_metadata, allow_pickle=False)
    
    if label is not None:
        np.save(os.path.join(target_directory, f"{example_id}_y.npy"), label, allow_pickle=False)
