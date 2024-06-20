import copy
import numpy as np
import pytorch_lightning as pl

from functools import partial

from monai.inferers import sliding_window_inference
from monai.utils.enums import MetricReduction

from .criterion_utils import DeepSupCriterion
from .metrics import DiceMetricWithEmptyLabels
from .postprocessing import ToDiscreteWithReplacingSmallET
from .utils import AverageMeter
from .loss import CriterionWrapper

class BaseTrainerModule(pl.LightningModule):
    def __init__(self, model, criterion, post_sigmoid, roi_size, infer_overlap, sw_batch_size):
        super().__init__()
        
        self.model = model
        self.model_inferer = partial(
            sliding_window_inference,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=self.model,
            overlap=infer_overlap,
        )
        
        self.criterion = CriterionWrapper(criterion)
        self.post_sigmoid = post_sigmoid
        
        self.acc_func = DiceMetricWithEmptyLabels(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
        self.post_pred = ToDiscreteWithReplacingSmallET(threshold=0.5, min_et=200, min_tc=0)
        self.run_acc = AverageMeter()
    
    def forward(self, x):
        return self.model.forward(x)
    
    def training_step(self, batch_data, batch_idx):
        data, target = batch_data['image'], batch_data['label']
        
        logits = self.forward(data)
        
        loss = self.criterion(logits, target)
        
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch_data, batch_idx):
        data, target = batch_data['image'], batch_data['label']
        
        logits = self.model_inferer(data)
        
        batch_loss = self.criterion(logits, target).cpu().item()
        
        heatmaps = self.post_sigmoid(logits)
        
        val_labels_list = [x for x in target]
        val_outputs_list = [x for x in heatmaps]
        val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        
        self.acc_func.reset()
        self.acc_func(y_pred=val_output_convert, y=val_labels_list)
        acc, not_nans = self.acc_func.aggregate()
        self.run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
        
        return {'sum_loss': len(data) * batch_loss, 'n_samples': len(data)}
    
    def val_test_epoch_end(self, set_name, step_outputs):
        total_loss = np.array([output['sum_loss'] for output in step_outputs]).sum()
        n_samples  = np.array([output['n_samples'] for output in step_outputs]).sum()
        
        dsc = copy.deepcopy(self.run_acc.avg)
        self.run_acc = AverageMeter()
        
        self.log(f"{set_name}_loss", total_loss / n_samples, prog_bar=True, logger=True)
        self.log(f"{set_name}_avg", np.mean(dsc), prog_bar=True, logger=True)
        self.log(f"{set_name}_tc", dsc[0], prog_bar=True, logger=True)
        self.log(f"{set_name}_wt", dsc[1], prog_bar=True, logger=True)
        self.log(f"{set_name}_et", dsc[2], prog_bar=True, logger=True)
    
    def validation_epoch_end(self, validation_step_outputs):
        return self.val_test_epoch_end("val", validation_step_outputs)
    
    def test_step(self, batch_data, batch_idx):
        return self.validation_step(batch_data, batch_idx)
    
    def test_epoch_end(self, test_step_outputs):
        return self.val_test_epoch_end("test", test_step_outputs)

class DeepSupBaseTrainerModule(BaseTrainerModule):
    def __init__(self, model, criterion, post_sigmoid, roi_size, infer_overlap, sw_batch_size,
                 deep_sup_levels, deep_sup_weights):
        super().__init__(model, criterion, post_sigmoid, roi_size, infer_overlap, sw_batch_size)
        
        self.criterion = DeepSupCriterion(criterion, deep_sup_levels, deep_sup_weights)
    
    def forward_features(self, x):
        return self.model.forward_features(x)
    
    def training_step(self, batch_data, batch_idx):
        data, target = batch_data['image'], batch_data['label']
        
        logits, deep_sup_logits = self.forward_features(data)
        
        loss = self.criterion(logits, target, deep_sup_logits, batch_data)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss