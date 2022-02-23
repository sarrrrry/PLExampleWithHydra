from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin

from pl_train import PROJECT_ROOT
from pl_train.dataloader.factory import DataLoaderFactory
from pl_train.pl_modules.mnist import MNISTModel


@dataclass
class ConfigTrain:
    LOGS_ROOT: str = "logs"
    RUN_NAME: str = "debug"
    PRJ_NAME: str = "pl_example"
    GROUP_NAME: str = "exp_1"

    debug: bool = True


cs = ConfigStore.instance()
cs.store(name="train", node=ConfigTrain)
hydra_dir = PROJECT_ROOT / "scripts" / "config" / "ml"


def build_loggers(cfg: ConfigTrain, now: str):
    from pytorch_lightning.loggers.wandb import WandbLogger
    from pytorch_lightning.loggers.csv_logs import CSVLogger

    hydra_dir = Path.cwd()
    name = f"{cfg.RUN_NAME}_{now}"

    loggers = [
        WandbLogger(save_dir=str(hydra_dir), name=name, project=cfg.PRJ_NAME, group=cfg.GROUP_NAME),
        CSVLogger(save_dir=str(hydra_dir / "csv"), name=name),
    ]
    return loggers


@hydra.main(config_path=hydra_dir, config_name="train")
def main(cfg: ConfigTrain):
    model = MNISTModel()
    now = Path.cwd().parts[-1]
    loggers = build_loggers(cfg, now=now)
    train_loader = DataLoaderFactory(
        train_root=Path("/media/hdd1/shared/pytorch/"),
        train_batchsize=5
    ).build()
    trainer = Trainer(
        gpus=-1, max_epochs=4, logger=loggers,
        # For fast https://pytorch-lightning.readthedocs.io/en/1.3.3/benchmarking/performance.html#
        strategy=DDPPlugin(find_unused_parameters=False),
        fast_dev_run=1 if cfg.debug else 0
    )
    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()
