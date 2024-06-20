import os
import json
import pandas as pd

def load_brats2023_fold(json_brats2021_fold, excel_brats_mapping):
    brats2021_to_brats2023 = {}
    for _, row in pd.read_excel(excel_brats_mapping).iterrows():
        if type(row.BraTS2023) != str:
            continue
        assert type(row.BraTS2021) == str
        brats2021_to_brats2023[row.BraTS2021] = row.BraTS2023
    
    with open(json_brats2021_fold) as file:
        brats2021 = json.load(file)
    
    brats2023 = {"training": []}
    for sample in brats2021['training']:
        name_2021 = os.path.dirname(sample['label'])
        name_2023 = brats2021_to_brats2023[name_2021]
        
        fold  = sample['fold']
        label = os.path.join(name_2023, f'{name_2023}-seg.nii.gz')
        image = [
            os.path.join(name_2023, f'{name_2023}-t2f.nii.gz'),
            os.path.join(name_2023, f'{name_2023}-t1c.nii.gz'),
            os.path.join(name_2023, f'{name_2023}-t1n.nii.gz'),
            os.path.join(name_2023, f'{name_2023}-t2w.nii.gz'),
        ]
        
        brats2023["training"].append({
            'fold' : fold,
            'image': image,
            'label': label,
        })
    
    return brats2023

def sanity_check_brats2023_data(dir_brats2023_train, brats2023_fold):
    brats2023_train = os.listdir(dir_brats2023_train)
    brats2023_train = set(brats2023_train)
    
    from_fold = [os.path.dirname(sample['label']) for sample in brats2023_fold['training']]
    from_fold = set(from_fold)
    
    assert len(brats2023_train) == len(from_fold)
    
    print(f'Number of training data: {len(brats2023_train)}')
    
    for sample_name in brats2023_train:
        assert sample_name in from_fold

def load_brats2024_goat_fold(json_brats2024_goat_fold):
    with open(json_brats2024_goat_fold) as file:
        brats2024 = json.load(file)
    
    brats2024_goat = {"training": []}
    for sample in brats2024['training']:
        name = os.path.dirname(sample['label'])
        
        fold  = sample['fold']
        label = os.path.join(name, f'{name}-seg.nii.gz')
        image = [
            os.path.join(name, f'{name}-t2f.nii.gz'),
            os.path.join(name, f'{name}-t1c.nii.gz'),
            os.path.join(name, f'{name}-t1n.nii.gz'),
            os.path.join(name, f'{name}-t2w.nii.gz'),
        ]
        
        brats2024_goat["training"].append({
            'fold' : fold,
            'image': image,
            'label': label,
        })
    
    return brats2024_goat

# Copied from
# https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb
def datafold_read(json_data, basedir, fold=0, key="training"):
    json_data = json_data[key]
    
    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
    
    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)
    
    return tr, val