import os
import sys
import json

import copy
import pytorch_lightning as pl
import schedulefree
import torch

from monai.data import DataLoader
from monai.transforms import Activations
from monai.utils import set_determinism

from pytorch_lightning.loggers import WandbLogger

from biomedmbz_glioma.pl_module import BaseTrainerModule
from biomedmbz_glioma.dataset import get_train_val_dataset
from biomedmbz_glioma.dataset_utils import load_brats2024_goat_fold
from biomedmbz_glioma.loss import get_loss_fn
from biomedmbz_glioma.callbacks import checkpoint_callback, lr_monitor, callback_save_last_only
from biomedmbz_glioma.scheduler import PolynomialLR, LinearWarmupCosineAnnealingWarmRestarts, LinearWarmupCosineAnnealingLR

from nnunet_mednext import create_mednext_v1
from nnunet_mednext.run.load_weights import upkern_load_weights

print("started_training")
 
def load_arguments_from_json(json_file):
    global args
    with open(json_file, 'r') as f:
        args_dict = json.load(f)
    args = Args(**args_dict)

class Args:
    is_debugging=False
    all_samples_as_train=False
    fold=0
    seed=42
    max_epochs=100
    preprocessed_data_dir="/home/abdelrahman.elsayed/Downloads/new_brats/BraTS2024_BioMedIAMBZ/dataset/pre-processed-images"
    json_brats2024_goat_fold= "/home/abdelrahman.elsayed/Downloads/new_brats/BraTS2024_BioMedIAMBZ/dataset/brats_ssa_2023_5_fold.json"
    mednext_size='B'
    mednext_ksize=3
    mednext_ckpt=None
    deep_sup=True
    batch_size=2
    sw_batch_size=4
    num_workers=4
    roi_x=128
    roi_y=128
    roi_z=128
    infer_overlap=0.5
    aug_type=1
    loss_type=3
    mean_batch=True
    lr=0.0027
    weight_decay=0
    lr_scheduler='free'
    n_gpus=1
    pin_memory=True
    souping = False # added this for souping option
    check_val_every_n_epoch=1
    wandb_project_name='brats-souping'

    def __init__(self, **kwargs):
        for kwarg in kwargs:
            setattr(Args, kwarg, kwargs[kwarg])
 
args = Args()
load_arguments_from_json("train_args.json")


torch.multiprocessing.set_sharing_strategy('file_system')
set_determinism(args.seed)

args.wandb_run_name = f'MedNeXt_fold-{args.fold}_' if args.all_samples_as_train == False \
    else f'MedNeXt_full-training_'
args.wandb_run_name += f'{args.max_epochs}-epochs_bs-{args.batch_size}_mednext-size-{args.mednext_size}_mednext-ksize-{args.mednext_ksize}_deep-sup-{args.deep_sup}_lr-{args.lr}_loss-type-{args.loss_type}_mean-batch-{args.mean_batch}_aug-type-{args.aug_type}_scheduler-{args.lr_scheduler}_seed-{args.seed}'

args.wandb_run_name = f'DEBUGGING-{args.wandb_run_name}' if args.is_debugging==True else args.wandb_run_name
wandb_logger = WandbLogger(
    name=args.wandb_run_name, project=args.wandb_project_name,
    config={f"{var_name}": f"{var_value}" for var_name, var_value in Args.__dict__.items() if not var_name.startswith('__')},
)

class TrainerModule(BaseTrainerModule):
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=args.lr,
            weight_decay=args.weight_decay,
        )
        
        if args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        elif args.lr_scheduler == 'multi-step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[int(args.max_epochs/3+0.5), int(args.max_epochs/3*2 + 0.5)], gamma=0.1,
            )
        elif args.lr_scheduler == 'cosine-with-warmup':
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=int(args.max_epochs/20 + 0.5), max_epochs=args.max_epochs,
                warmup_start_lr=1e-7, eta_min=1e-6,
            )
        elif args.lr_scheduler == 'cosine-with-warmup-with-restarts':
            scheduler = LinearWarmupCosineAnnealingWarmRestarts(
                optimizer, max_epochs=args.max_epochs, n_cycles=3,
                warmup_start_lr=1e-8, eta_min=1e-8,
                gamma=0.5, pct_warmup_epoch=0.1,
            )
        elif args.lr_scheduler == 'polynomial':
            scheduler = PolynomialLR(
                optimizer, total_iters=args.max_epochs, power=0.9,
            )
        elif args.lr_scheduler == 'free':
            optimizer = schedulefree.AdamWScheduleFree(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = None
            return [optimizer]
        else:
            raise ValueError(f"Learning rate scheduler '{args.lr_scheduler}' is not recognized ...")
        
        return [optimizer], [scheduler]

class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        
        brats2024 = load_brats2024_goat_fold(args.json_brats2024_goat_fold)
        
        roi_size = (args.roi_x, args.roi_y, args.roi_z)
        self.train_dataset, self.val_dataset = get_train_val_dataset(
            args.preprocessed_data_dir, brats2024, args.fold, roi_size, args.aug_type,
            args.all_samples_as_train,
        )
        self.test_dataset = copy.deepcopy(self.val_dataset)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True,
            pin_memory=args.pin_memory,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=1, shuffle=False,
            num_workers=args.num_workers, drop_last=False,
            pin_memory=args.pin_memory,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=1, shuffle=False,
            num_workers=args.num_workers, drop_last=False,
            pin_memory=args.pin_memory,
        )

dm = DataModule()

model = create_mednext_v1(
    num_input_channels=4,
    num_classes=3,
    model_id=args.mednext_size,
    kernel_size=args.mednext_ksize,
    deep_supervision=args.deep_sup,
    checkpoint_style='outside_block',
)
if args.mednext_ckpt:
    assert args.mednext_ksize == 5
    _model = create_mednext_v1(
        num_input_channels=4,
        num_classes=3,
        model_id=args.mednext_size,
        kernel_size=3,
        deep_supervision=args.deep_sup,
        checkpoint_style='outside_block',
    )
    ckpt = torch.load(args.mednext_ckpt, map_location='cpu')
    for key in list(ckpt['state_dict'].keys()):
        if key.startswith('model.'):
            ckpt['state_dict'][key[6:]] = ckpt['state_dict'].pop(key)
    _model.load_state_dict(ckpt['state_dict'])
    
    model = upkern_load_weights(model, _model)

module = TrainerModule(
    model, get_loss_fn(args.loss_type, args.mean_batch), Activations(sigmoid=True),
    [args.roi_x, args.roi_y, args.roi_z],
    args.infer_overlap, args.sw_batch_size,
)

list_callbacks = [checkpoint_callback, lr_monitor] if args.all_samples_as_train == False \
    else [callback_save_last_only, lr_monitor]

trainer = pl.Trainer(
    gpus=args.n_gpus, max_epochs=args.max_epochs,
    check_val_every_n_epoch=args.check_val_every_n_epoch,
    log_every_n_steps=10,
    callbacks=list_callbacks,
    logger= wandb_logger,
    precision=32,
    amp_backend='native',
    benchmark=True,
    limit_val_batches=1.0 if args.all_samples_as_train == False else 0,
    num_sanity_val_steps=2 if args.all_samples_as_train == False else 0,
)

trainer.fit(module, dm)
if args.all_samples_as_train == False:
    trainer.test(module, dataloaders=dm.test_dataloader(), ckpt_path='best')

print("done_training")