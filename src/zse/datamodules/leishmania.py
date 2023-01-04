from abc import ABC
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from zse.datamodules.components.leishmania_dataset import LeishmaniaDataset


class LeishmaniaDataModule(LightningDataModule, ABC):
    def __init__(
            self,
            data_dir: str = "data/",
            imsize: int = None,
            scale: float = None,
            batch_size: int = 4,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.transforms = T.Compose([
            T.ToTensor(),
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_test:
            self.data_train = LeishmaniaDataset(
                self.hparams.data_dir,
                imsize=self.hparams.imsize,
                scale=self.hparams.scale,
                transform=self.transforms)
            self.data_test = LeishmaniaDataset(
                self.hparams.data_dir.replace("train", "test"),
                imsize=self.hparams.imsize,
                scale=self.hparams.scale,
                transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True)

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False)
