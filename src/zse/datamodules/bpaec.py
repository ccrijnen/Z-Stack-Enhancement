from abc import ABC
from copy import copy
from typing import Optional, Union, Type

import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from zse.datamodules.components.bpaec_dataset import BPAEC2D, BPAEC3D


class BPAECDataModule(LightningDataModule, ABC):
    def __init__(
            self,
            dataset_cls: Type[Union[BPAEC2D, BPAEC3D]],
            data_dir: str = "data/",
            dset: str = "",
            file_glob: str = "/*/*.jpg",
            train_test_split: float = 0.8,
            resize_size: int = None,
            batch_size: int = 4,
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.kwargs = kwargs

        self.transforms = T.Compose([
            T.ToTensor(),
            # T.RandomCrop(512),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply([T.RandomRotation((90, 90))], p=0.5),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_test:
            if self.hparams.dset == "all":
                dset_train = []
                dset_test = []
                for d in ["nucleus", "actin", "mitochondria"]:
                    dataset = self.hparams.dataset_cls(
                        self.hparams.data_dir + d + self.hparams.file_glob,
                        transform=self.transforms,
                        resize_size=self.hparams.resize_size,
                        **self.kwargs
                    )
                    indices = range(len(dataset))
                    split_ind = round(self.hparams.train_test_split * len(dataset))
                    dset_train.append(Subset(dataset, indices[:split_ind]))
                    test_dset = Subset(copy(dataset), indices[split_ind:])
                    test_dset.dataset.transform = T.ToTensor()
                    dset_test.append(test_dset)
                self.data_train = ConcatDataset(dset_train)
                self.data_test = ConcatDataset(dset_test)
            else:
                dataset = self.hparams.dataset_cls(
                    self.hparams.data_dir + self.hparams.dset + self.hparams.file_glob,
                    transform=self.transforms,
                    resize_size=self.hparams.resize_size,
                    **self.kwargs
                )
                indices = range(len(dataset))
                split_ind = round(self.hparams.train_test_split * len(dataset))
                self.data_train = Subset(dataset, indices[:split_ind])
                self.data_test = Subset(copy(dataset), indices[split_ind:])
                self.data_test.dataset.transform = T.ToTensor()

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


class BPAECDataModule2D(BPAECDataModule, ABC):
    def __init__(
            self,
            data_dir: str = "data/",
            dset: str = "",
            file_glob: str = "/*/*.jpg",
            train_test_split: float = 0.8,
            resize_size: int = None,
            batch_size: int = 4,
            num_workers: int = 0,
            pin_memory: bool = False
    ):
        super().__init__(dataset_cls=BPAEC2D, data_dir=data_dir, dset=dset, file_glob=file_glob,
                         train_test_split=train_test_split, resize_size=resize_size, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=pin_memory)


class BPAECDataModule3D(BPAECDataModule, ABC):
    def __init__(
            self,
            data_dir: str = "data/",
            dset: str = "",
            file_glob: str = "/*/*.jpg",
            train_test_split: float = 0.8,
            resize_size: int = None,
            pad: int = None,
            batch_size: int = 4,
            num_workers: int = 0,
            pin_memory: bool = False
    ):
        super().__init__(dataset_cls=BPAEC3D, data_dir=data_dir, dset=dset, file_glob=file_glob,
                         train_test_split=train_test_split, resize_size=resize_size, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=pin_memory, pad=pad)
