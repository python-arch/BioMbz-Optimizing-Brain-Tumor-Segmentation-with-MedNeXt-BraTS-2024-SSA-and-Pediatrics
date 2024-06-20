import torch

from monai.metrics import DiceMetric
from monai.metrics.meandice import compute_dice
from monai.metrics.utils import is_binary_tensor

class DiceMetricWithEmptyLabels(DiceMetric):
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        is_binary_tensor(y_pred, "y_pred")
        is_binary_tensor(y, "y")
        
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        # compute dice (BxC) for each channel for each batch
        acc = compute_dice(
            y_pred=y_pred, y=y, include_background=self.include_background, ignore_empty=self.ignore_empty
        )
        
        for n in range(acc.shape[0]):
            for c in range(acc.shape[1]):
                if y[n][c].sum() != 0:
                    continue
                acc[n][c] = 1.0 if y_pred[n][c].sum() == 0 else 0.0
        
        return acc