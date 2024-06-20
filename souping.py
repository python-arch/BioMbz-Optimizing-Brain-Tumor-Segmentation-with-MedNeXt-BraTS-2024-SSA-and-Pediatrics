import os
import sys

import copy
import pytorch_lightning as pl
import schedulefree
import torch

from monai.data import DataLoader
from monai.transforms import Activations


from biomedmbz_glioma.pl_module import BaseTrainerModule
from biomedmbz_glioma.dataset import get_train_val_dataset
from biomedmbz_glioma.dataset_utils import load_brats2024_goat_fold
from biomedmbz_glioma.loss import get_loss_fn
from biomedmbz_glioma.callbacks import checkpoint_callback, lr_monitor, callback_save_last_only
from biomedmbz_glioma.scheduler import PolynomialLR, LinearWarmupCosineAnnealingWarmRestarts, LinearWarmupCosineAnnealingLR

from nnunet_mednext import create_mednext_v1

import model_soup

class Args:
    is_debugging=False
    all_samples_as_train=False
    fold=2
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
    check_val_every_n_epoch=1
    greedy = False # added this for souping option
 
args = Args()

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


# callbacks for the trainer
list_callbacks = [checkpoint_callback, lr_monitor] if args.all_samples_as_train == False \
    else [callback_save_last_only, lr_monitor]

# the trainer
trainer = pl.Trainer(
    gpus=args.n_gpus, max_epochs=args.max_epochs,
    check_val_every_n_epoch=args.check_val_every_n_epoch,
    log_every_n_steps=10,
    callbacks=list_callbacks,
    precision=32,
    amp_backend='native',
    benchmark=True,
    limit_val_batches=1.0 if args.all_samples_as_train == False else 0,
    num_sanity_val_steps=2 if args.all_samples_as_train == False else 0,
)

def get_test_avg(model):
    module = TrainerModule(
    model, get_loss_fn(args.loss_type, args.mean_batch), Activations(sigmoid=True),
    [args.roi_x, args.roi_y, args.roi_z],
    args.infer_overlap, args.sw_batch_size,
)
    metrics = trainer.test(module , dataloaders=dm.test_dataloader())[0]
    return metrics['test_avg']

# get the ckpt files from the souping models dir
def find_ckpt_files(base_path):
    ckpt_paths = []
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(base_path):
        # Check if 'checkpoints' is in the current directory
        if 'checkpoints' in dirs:
            checkpoints_dir = os.path.join(root, 'checkpoints')
            # List all files in the checkpoints directory
            for file_name in os.listdir(checkpoints_dir):
                if file_name.endswith('.ckpt') and file_name !="last.ckpt":
                    ckpt_paths.append(os.path.join(checkpoints_dir, file_name))
                
                    
    return ckpt_paths


base_path = f'/home/abdelrahman.elsayed/Downloads/final_souping_models/models/fold {args.fold}'
ckpt_file_paths = find_ckpt_files(base_path)


# load the models from the checkpoint given
def load_model(checkpoint_path):
    model = create_mednext_v1(
                num_input_channels=4,
                num_classes=3,
                model_id=args.mednext_size,
                kernel_size=3,
                deep_supervision=args.deep_sup,
                checkpoint_style='outside_block',
            )
    ckpt = torch.load(checkpoint_path, map_location='cuda')
    for key in list(ckpt['state_dict'].keys()):
                if key.startswith('model.'):
                    ckpt['state_dict'][key[6:]] = ckpt['state_dict'].pop(key)
    model.load_state_dict(ckpt['state_dict'])

    return model

# scores of the models
scored_checkpoints = []
# sort the list of checkpoints desending after loading each model and test it
def sort_checkpoints_by_score(ckpt_paths):
    
    for ckpt_path in ckpt_paths:
        model = load_model(ckpt_path)
        score = get_test_avg(model)
        scored_checkpoints.append((ckpt_path, score))
    
    # Sort by score in descending order
    scored_checkpoints.sort(key=lambda x: x[1], reverse=True)
    
    # Extract the sorted paths
    sorted_paths = [ckpt_path for ckpt_path, _ in scored_checkpoints]
    
    return sorted_paths

sorted_paths = sort_checkpoints_by_score(ckpt_file_paths)

# create mednext Souped model
best_model = load_model(sorted_paths[0])
if args.greedy:
    print(f'Doing Greedy Soup for fold {args.fold}')
    model_souped = model_soup.torch.greedy_souping(best_model, sorted_paths , metric=get_test_avg , device="cuda" if args.n_gpus > 0 else "cpu")
else:
    print(f"Doing Uniform Soup for fold {args.fold}")
    model_souped = model_soup.torch.uniform_soup(best_model, sorted_paths,device='cuda')

module_souped = TrainerModule(
    model_souped, get_loss_fn(args.loss_type, args.mean_batch), Activations(sigmoid=True),
    [args.roi_x, args.roi_y, args.roi_z],
    args.infer_overlap, args.sw_batch_size,
)

best_module = TrainerModule(
    best_model, get_loss_fn(args.loss_type, args.mean_batch), Activations(sigmoid=True),
    [args.roi_x, args.roi_y, args.roi_z],
    args.infer_overlap, args.sw_batch_size,
)
print(f"The metric for the individual model:")
trainer.test(best_module , dataloaders=dm.test_dataloader())
type = "Greedy" if args.greedy else "Uniform"
print(f"The metrics for the {type} souped three models")
trainer.test(module_souped , dataloaders=dm.test_dataloader())