import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

def get_region(value, milestones):
    for i, milestone in enumerate(milestones):
        if value <= milestone-1:
            break
    return i

class LinearWarmupCosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_epochs: int,
        pct_warmup_epoch: float,
        n_cycles: int = 1,
        gamma: float = 0.1,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if pct_warmup_epoch <= 0 or pct_warmup_epoch >= 1. or not isinstance(pct_warmup_epoch, float):
            raise ValueError("Expected float 0 < pct_warmup_epoch < 1, but got {}".format(pct_warmup_epoch))
        
        self.pct_warmup_epoch = pct_warmup_epoch
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.gamma = gamma
        self.n_cycles = n_cycles
        
        self.cycle_length = int(self.max_epochs / self.n_cycles)
        self.milestones = [int((i + 1) * self.cycle_length) for i in range(self.n_cycles)]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )
        
        region = get_region(self.last_epoch, self.milestones)
        base_lrs = [base_lr * self.gamma**region for base_lr in self.base_lrs]
        warmup_epochs = int(self.max_epochs * self.pct_warmup_epoch / self.n_cycles)
        last_epoch = self.last_epoch - region * self.cycle_length
        
        if last_epoch == warmup_epochs:
            return base_lrs
        if last_epoch == 0:
            return [self.warmup_start_lr] * len(base_lrs)
        if last_epoch < warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (warmup_epochs - 1)
                for base_lr, group in zip(base_lrs, self.optimizer.param_groups)
            ]
        if (last_epoch - 1 - self.cycle_length) % (2 * (self.cycle_length - warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.cycle_length - warmup_epochs))) / 2
                for base_lr, group in zip(base_lrs, self.optimizer.param_groups)
            ]
        
        return [
            (1 + math.cos(math.pi * (last_epoch - warmup_epochs) / (self.cycle_length - warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (last_epoch - warmup_epochs - 1) / (self.cycle_length - warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
    
    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        raise NotImplementedError

class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]
        
        decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self):
        return [
            (
                base_lr * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
            )
            for base_lr in self.base_lrs
        ]

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                UserWarning,
            )

        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch * (base_lr - self.warmup_start_lr) / max(1, self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]