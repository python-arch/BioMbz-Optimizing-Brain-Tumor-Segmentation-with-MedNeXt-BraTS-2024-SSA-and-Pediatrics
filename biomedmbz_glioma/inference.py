import os
import copy
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn

from functools import partial
from pathlib import Path

# from threading import Thread

from monai.inferers import sliding_window_inference
# from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Activations
from monai.utils.enums import MetricReduction

from .utils import AverageMeter
from .metrics import DiceMetricWithEmptyLabels

def to_npz(path, label, output):
    np.savez(path, label=label, output=output)

class InferenceModule(pl.LightningModule):
    def __init__(
        self,
        model,
        apply_sigmoid,
        roi_size,
        infer_overlap,
        sw_batch_size,
        post_pred=AsDiscrete(argmax=False, threshold=0.5),
        save_statistics=False,
        eval_name=None,
        fold=None,
        tta=False,
        blend='constant',
    ):
        super().__init__()
        
        self.model = model
        self.tta = tta
        
        self.model_inferer = partial(
            sliding_window_inference,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=self.model,
            overlap=infer_overlap,
            mode=blend,
        )
        
        self.apply_sigmoid = apply_sigmoid
        self.fn_sigmoid    = Activations(sigmoid=True)
        self.post_pred     = post_pred if post_pred is not None else AsDiscrete(argmax=False, threshold=0.5)
        
        # self.acc_func  = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
        self.acc_func  = DiceMetricWithEmptyLabels(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
        self.run_acc   = AverageMeter()
        
        self.save_statistics = save_statistics
        if save_statistics == True:
            self.list_statistics = []
            self.list_num_pixels = []
            # self.threads = []
            self.eval_name = eval_name
            self.fold = fold
            Path(self.eval_name).mkdir(parents=True, exist_ok=True)
            # Path(os.path.join(self.eval_name, str(self.fold))).mkdir(parents=True, exist_ok=True)
    
    def forward(self, image):
        if len(image.shape) == 4:
            image = image.unsqueeze(0)
        elif len(image.shape) != 5:
            raise ValueError(f"The input image size of '{image.shape}' is not compatible ...")
        
        preds = self.model_inferer(image)
        if self.tta == True:
            flip_combs = [[2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]]
            for dims in flip_combs:
                preds += torch.flip(self.model_inferer(torch.flip(image, dims)), dims)
            preds = preds / (1. + len(flip_combs))
        
        if self.apply_sigmoid == True:
            preds = self.fn_sigmoid(preds)
        
        outs = [self.post_pred(pred).cpu() for pred in preds]
        
        return outs
    
    def test_step(self, batch_data, batch_idx):
        data, target = batch_data['image'], batch_data['label']
        
        preds = self.model_inferer(data)
        if self.tta == True:
            flip_combs = [[2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]]
            for dims in flip_combs:
                preds += torch.flip(self.model_inferer(torch.flip(data, dims)), dims)
            preds = preds / (1. + len(flip_combs))
        
        if self.apply_sigmoid == True:
            preds = self.fn_sigmoid(preds)
        
        val_labels_list = [x for x in target]
        val_outputs_list = [x for x in preds]
        val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        
        self.acc_func.reset()
        self.acc_func(y_pred=val_output_convert, y=val_labels_list)
        acc, not_nans = self.acc_func.aggregate()
        self.run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
        
        if self.save_statistics:
            acc = acc.cpu().numpy()
            filename = Path(batch_data['image_meta_dict']['filename_or_obj'][0]).parts[-2]
            self.list_statistics.append([filename, acc[0], acc[1], acc[2], np.mean(acc)])
            val_output = np.where(val_output_convert[0].cpu().numpy() >= 0.5, True, False).astype(np.bool)
            val_label  = val_labels_list[0].cpu().numpy()
            path = os.path.join(self.eval_name, str (self.fold), f'{filename}.npz')
            
            self.list_num_pixels.append([
                filename, np.sum(val_label[0]==1), np.sum(val_label[1]==1), np.sum(val_label[2]==1),
                np.sum(val_output[0]==1), np.sum(val_output[1]==1), np.sum(val_output[2]==1),
            ])
            
            # thread = Thread(target=to_npz, args=(path, val_label, val_output))
            # thread.start()
            # self.threads.append(thread)
    
    def test_epoch_end(self, step_outputs):
        set_name = 'test'
        
        dsc = copy.deepcopy(self.run_acc.avg)
        self.run_acc = AverageMeter()
        
        self.log(f"{set_name}_avg", np.mean(dsc), prog_bar=True, logger=True)
        self.log(f"{set_name}_tc", dsc[0], prog_bar=True, logger=True)
        self.log(f"{set_name}_wt", dsc[1], prog_bar=True, logger=True)
        self.log(f"{set_name}_et", dsc[2], prog_bar=True, logger=True)
        
        if self.save_statistics:
            df = pd.DataFrame(self.list_statistics, columns=['filename', 'tc', 'wt', 'et', 'avg'])
            df.to_csv(os.path.join(self.eval_name, f'statistics_fold-{self.fold}.csv'))
            
            df = pd.DataFrame(self.list_num_pixels, columns=['filename', 'gt_tc', 'gt_wt', 'gt_et', 'pr_tc', 'pr_wt', 'pr_et'])
            df.to_csv(os.path.join(self.eval_name, f'num_pixels_fold-{self.fold}.csv'))
            
            self.list_statistics = []
            self.list_num_pixels = []
            # for thread in self.threads:
            #     thread.join()
        
        return {
            'avg': np.mean(dsc),
            'tc' : dsc[0],
            'wt' : dsc[1],
            'et' : dsc[2],
        }