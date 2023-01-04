from abc import ABC
from copy import copy
from typing import Optional

import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from zse.datamodules.components.brain_datasets import ZStackDataset2D, ZStackDataset3D
from zse.utils.data_utils import get_transform


class BrainDataModule(LightningDataModule, ABC):
    def __init__(
            self,
            imsize: int = 256,
            random_crop: bool = False,
            batch_size: int = 4,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        self.transforms_train = T.Compose([
            get_transform(imsize, random_crop),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply([T.RandomRotation((90, 90))], p=0.5),
        ])
        self.transforms_test = get_transform(imsize, False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

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


class BrainDataModule2D(BrainDataModule, ABC):
    def __init__(
            self,
            data_dir_train: str = "data/",
            val_length: int = 2000,
            resample: bool = False,
            z_target: int = 29,
            z_depth: int = 29,
            binary: bool = False,
            imsize: int = 256,
            random_crop: bool = True,
            batch_size: int = 4,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__(imsize=imsize, random_crop=random_crop, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=pin_memory)
        self.save_hyperparameters(logger=False)
        self.hparams.val_length = val_length * (z_target + int(binary))

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = ZStackDataset2D(
                self.hparams.data_dir_train,
                transform=self.transforms_train,
                resample=self.hparams.resample,
                z_target=self.hparams.z_target,
                z_depth=self.hparams.z_depth,
                binary=self.hparams.binary,
            )
            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=(len(dataset)-self.hparams.val_length, self.hparams.val_length),
                generator=torch.Generator().manual_seed(42),
            )
            self.data_val.dataset = copy(dataset)
            self.data_val.dataset.transform = self.transforms_test
            self.data_test = ZStackDataset2D(
                self.hparams.data_dir_train.replace("train", "test"),
                transform=self.transforms_test,
                resample=self.hparams.resample,
                z_target=self.hparams.z_target,
                z_depth=self.hparams.z_depth,
                binary=self.hparams.binary,
            )


class BrainDataModule3D(BrainDataModule, ABC):
    def __init__(
            self,
            data_dir_train: str = "data/",
            val_length: int = 2000,
            resample: bool = False,
            z_target: int = 29,
            imsize: int = 256,
            random_crop: bool = True,
            batch_size: int = 4,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__(imsize=imsize, random_crop=random_crop, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=pin_memory)
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = ZStackDataset3D(
                self.hparams.data_dir_train,
                transform=self.transforms_train,
                resample=self.hparams.resample,
                z_target=self.hparams.z_target,
            )
            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=(len(dataset) - self.hparams.val_length, self.hparams.val_length),
                generator=torch.Generator().manual_seed(42),
            )
            self.data_val.dataset = copy(dataset)
            self.data_val.dataset.transform = self.transforms_test
            self.data_test = ZStackDataset3D(
                self.hparams.data_dir_train.replace("train", "test"),
                transform=self.transforms_test,
                resample=self.hparams.resample,
                z_target=self.hparams.z_target,
            )
