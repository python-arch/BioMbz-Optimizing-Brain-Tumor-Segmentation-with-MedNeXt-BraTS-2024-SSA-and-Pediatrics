import os

from monai import data

from pathlib import Path

from .dataset_utils import datafold_read
from .transforms import get_train_val_transforms, get_test_transform
from .transforms import deep_sup_get_train_val_transforms

def get_train_val_dataset(preprocessed_data_dir, dict_fold_brats, fold, roi_size, aug_type=1, all_samples_as_train=False):
    train_files, validation_files = datafold_read(json_data=dict_fold_brats, basedir='.', fold=fold)
    
    if all_samples_as_train == True:
        train_files.extend(validation_files)
    
    train_files      = [{
        'name' : Path(sample['image'][0]).parts[-2],
        'image': os.path.join(preprocessed_data_dir, f"{Path(sample['image'][0]).parts[-2]}_x.npy"),
        'label': os.path.join(preprocessed_data_dir, f"{Path(sample['image'][0]).parts[-2]}_y.npy"),
        'meta' : os.path.join(preprocessed_data_dir, f"{Path(sample['image'][0]).parts[-2]}_meta.npy"),
    } for sample in train_files]
    validation_files =  [{
        'name' : Path(sample['image'][0]).parts[-2],
        'image': os.path.join(preprocessed_data_dir, f"{Path(sample['image'][0]).parts[-2]}_x.npy"),
        'label': os.path.join(preprocessed_data_dir, f"{Path(sample['image'][0]).parts[-2]}_y.npy"),
        'meta' : os.path.join(preprocessed_data_dir, f"{Path(sample['image'][0]).parts[-2]}_meta.npy"),
    } for sample in validation_files]
    
    train_transform, val_transform = get_train_val_transforms(roi_size, aug_type)
    
    train_dataset = data.Dataset(data=train_files, transform=train_transform)
    val_dataset   = data.Dataset(data=validation_files, transform=val_transform)
    
    return train_dataset, val_dataset

def get_test_dataset(data_dir):
    test_transform = get_test_transform()
    
    test_files = []
    for f in os.listdir(data_dir):
        if '_meta.npy' in f:
            continue
        name = f.split('.')[0][:-2]
        test_files.append({
            'name' : name,
            'image': os.path.join(data_dir, f'{name}_x.npy'),
            'meta' : os.path.join(data_dir, f'{name}_meta.npy'),
        })
    
    test_dataset = data.Dataset(data=test_files, transform=test_transform)
    
    return test_dataset

def deep_sup_get_train_val_dataset(data_dir, dict_fold_brats, fold, roi_size, deep_sup_levels):
    train_files, validation_files = datafold_read(json_data=dict_fold_brats, basedir=data_dir, fold=fold)
    
    train_transform, val_transform = deep_sup_get_train_val_transforms(roi_size, deep_sup_levels)
    
    train_dataset = data.Dataset(data=train_files, transform=train_transform)
    val_dataset   = data.Dataset(data=validation_files, transform=val_transform)
    
    return train_dataset, val_dataset

def c_get_train_val_dataset(data_dir, dict_fold_brats, fold, roi_size):
    train_files, validation_files = datafold_read(json_data=dict_fold_brats, basedir=data_dir, fold=fold)
    
    train_transform, val_transform = get_train_val_transforms(roi_size)
    
    train_dataset = data.Dataset(data=train_files, transform=train_transform)
    val_dataset   = data.CacheDataset(
        data=validation_files, transform=val_transform,
        cache_rate=1.0, runtime_cache=None, copy_cache=False,
        num_workers=8, 
    )
    
    return train_dataset, val_dataset