from abc import ABC, abstractmethod
from typing import cast

import torch.optim
from pytorch_lightning import LightningModule

from pl_train.optimizers.sam import SAM


class BaseWithSAM(LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        optimizer = cast(SAM, self.optimizers())

        # 1st forward-backward pass
        loss_1 = self._compute_loss(batch)
        self.manual_backward(loss_1)
        optimizer.first_step(zero_grad=True)

        # 2nd forward-backward pass
        loss_2 = self._compute_loss(batch)
        self.manual_backward(loss_2)
        optimizer.second_step(zero_grad=False)

        return loss_2

    @abstractmethod
    def _compute_loss(self, batch) -> torch.Tensor:
        ...

    def configure_optimizers(self):
        return SAM(self.parameters(), torch.optim.SGD, rho=0.05, adaptive=False, lr=0.1, momentum=0.9)