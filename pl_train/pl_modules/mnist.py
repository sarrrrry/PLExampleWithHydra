import torch.nn
from torch.nn import functional as F

from pl_train.pl_modules.base_with_sam import BaseWithSAM


class MNISTModel(BaseWithSAM):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)

        self.log("loss/train", loss.item(), prog_bar=True)

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
