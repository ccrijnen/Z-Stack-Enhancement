from abc import ABC
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from zse.datamodules.components.brain_volume_dataset import ZStackVolDataset


class BrainVolumeDataModule(LightningDataModule, ABC):
    def __init__(
            self,
            data_dir: str = "/p/fastdata/bigbrains/personal/schiffer1/experiments/2022_bigbrain_1micron/" +
                            "data/kg/vol*/data/BigBrain_VOI_*_02um_tiled/*/*.hdf5",
            destination: str = "/p/fastdata/bigbrains/personal/crijnen1/data/christian",
            train_val_test_split: Tuple[int, int, int] = (265, 16, 31),
            batch_size: int = 4,
            num_workers: int = 32,
            pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_test:
            dataset = ZStackVolDataset(self.hparams.data_dir, self.hparams.destination)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
