from pathlib import Path
class DataLoaderFactory:
    def __init__(self, train_root: Path, train_batchsize: int = 5):
        self.train_root = train_root
        self.train_batchsize = train_batchsize

    def build(self):
        from torchvision.datasets import MNIST
        from torchvision import transforms
        from torch.utils.data import DataLoader
        import os
        train_ds = MNIST(
            str(self.train_root),
            train=True, download=True,
            transform=transforms.ToTensor()
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.train_batchsize,
            num_workers=os.cpu_count()
        )
        return train_loader