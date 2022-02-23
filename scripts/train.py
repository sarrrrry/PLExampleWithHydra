from abc import abstractmethod, ABC
from typing import cast

import torch.nn
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F

from pl_train.optimizers.sam import SAM


class BaseWithSAM(LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        optimizer = cast(SAM, self.optimizers())

        # 1st forward-backward pass
        loss_1 = self._compute_loss(batch)
        self.manual_backward(loss_1)
        optimizer.first_step(zero_grad=True)

        # 2nd forward-backward pass
        loss_2 = self._compute_loss(batch)
        self.manual_backward(loss_2)
        optimizer.second_step(zero_grad=True)

        return loss_2

    @abstractmethod
    def _compute_loss(self, batch):
        ...

    def configure_optimizers(self):
        return SAM(self.parameters(), torch.optim.SGD, rho=0.05, adaptive=False, lr=0.1, momentum=0.9)

class MNISTModel(BaseWithSAM):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.l1(x)
        x = torch.relu(x)
        return x

    def _compute_loss(self, batch):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        return loss


class DataLoaderFactory:
    def __init__(self, train_batchsize: int = 5):
        self.train_batchsize = train_batchsize

    def build(self):
        from torchvision.datasets import MNIST
        from torchvision import transforms
        from torch.utils.data import DataLoader
        import os
        train_ds = MNIST(
            "/media/shared/pytorch/",
            train=True, download=True,
            transform=transforms.ToTensor()
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.train_batchsize,
            num_workers=os.cpu_count()
        )
        return train_loader


model = MNISTModel()
train_loader = DataLoaderFactory(train_batchsize=5).build()
trainer = Trainer(gpus=0, max_epochs=3)
trainer.fit(model, train_loader)
