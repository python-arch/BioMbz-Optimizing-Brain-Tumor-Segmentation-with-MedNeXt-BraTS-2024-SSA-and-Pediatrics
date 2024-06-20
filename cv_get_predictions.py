import os
import torch
import torch.nn as nn
import numpy as np
import shutil

from pathlib import Path
from threading import Thread
from tqdm import tqdm

from monai.data import DataLoader
from monai.utils import set_determinism

from biomedmbz_glioma.inference import InferenceModule
from biomedmbz_glioma.dataset import get_train_val_dataset
from biomedmbz_glioma.dataset_utils import load_brats2024_goat_fold

from model_directory import *

class Args:
    prefix='MedNeXt_B3'
    model_cls = get_model_mednext_b3 # Check model_directory.py (e.g. get_model_mednext_b3 if you use MedNeXt-B with k=3)
    dict_checkpoint = { # Checkpoints from training
        0: '... .ckpt',
        1: '... .ckpt',
        2: '... .ckpt',
        3: '... .ckpt',
        4: '... .ckpt',
    }
    infer_overlap=0.5
    aggregate_level="probability"
    tta=True
    save_dir = ......................
    eval_name = f'{prefix}_infer-overlap-{infer_overlap}_agg-lvl-{aggregate_level}_tta-{tta}'
    sw_batch_size=4
    num_workers=2
    preprocessed_data_dir = ......................
    json_brats2024_goat_fold = ......................
    roi_x=128
    roi_y=128
    roi_z=128
    cuda=1
    pin_memory=False
    seed=42

args = Args()
torch.multiprocessing.set_sharing_strategy('file_system')

def save_pred_to_npy(path, pred):
    np.save(path, pred)

def get_fold_prediction(fold, ckpt_path, save_dir):
    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    
    set_determinism(args.seed)
    
    brats2024 = load_brats2024_goat_fold(args.json_brats2024_goat_fold)
    
    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    _, val_dataset = get_train_val_dataset(
        args.preprocessed_data_dir, brats2024, fold, roi_size,
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
        pin_memory=args.pin_memory,
    )
    
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    model  = args.model_cls()
    module = InferenceModule(model, not model.apply_sigmoid, roi_size, args.infer_overlap, args.sw_batch_size, nn.Identity(), tta=args.tta)
    
    module.model.load_state_dict(checkpoint['state_dict'])
    module.eval()
    
    if args.cuda: module.cuda()
    
    threads = []
    for sample in tqdm(val_dataloader):
        with torch.no_grad():
            preds = module(sample["image"].cuda()) if args.cuda else module(sample["image"])
            
            for name_2023, pred in zip(sample['name'], preds):
                thread = Thread(target=save_pred_to_npy, args=(os.path.join(save_dir, f'{name_2023}.npy'), pred.numpy(),))
                thread.start()
                threads.append(thread)
    
    if args.cuda: module.cpu()
    
    for thread in threads:
        thread.join()

for fold in sorted(args.dict_checkpoint.keys()):
    print(f'FOLD: {fold}')
    save_dir = os.path.join(args.save_dir, args.eval_name, str(fold))
    
    if os.path.exists(save_dir) and os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=False)
    
    ckpt_path = args.dict_checkpoint[fold]
    get_fold_prediction(fold, ckpt_path, save_dir)


# scp -r fadillah.maani@login-student-lab.mbzu.ae:/home/fadillah.maani/BraTS-GOAT/BraTS2023-Glioma/brats2024-goat/rlh1r9hn m-b_k-3_f-1

