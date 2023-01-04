import glob
import itertools

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def sort_fn(x: str):
    split = x.split('_')
    z = int(split[-2].split('/')[-2][-2:])
    n = int(split[-1].split('.')[0])
    return int(f"{n:02}{z:02}")


def group_fn(x: str):
    return int(x.split('_')[-1].split('.')[0])


class BPAECDataset(Dataset):
    def __init__(self, data_glob, transform, resize_size=None):
        self.file_names = sorted(glob.glob(data_glob), key=sort_fn)
        self.transform = transform
        self.resize_size = resize_size


class BPAEC2D(BPAECDataset):
    def __init__(self, data_glob, transform, resize_size=None):
        super().__init__(data_glob=data_glob, transform=transform, resize_size=resize_size)
        self.file_names = [f for f in self.file_names if "Z7" not in f]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path_lq = self.file_names[idx]
        z = int(path_lq.split('/')[-2][-2:])
        path_hq = path_lq.replace(f"Z0{z:02}", "Z007").replace(f"Z{z}", "Z7")
        lq = cv2.imread(path_lq, 0)
        hq = cv2.imread(path_hq, 0)
        if self.resize_size is not None:
            lq = cv2.resize(lq, dsize=(self.resize_size, self.resize_size))
            hq = cv2.resize(hq, dsize=(self.resize_size, self.resize_size))
        comb = np.stack([lq, hq], axis=2)
        comb_t = self.transform(comb)
        lq, hq = torch.split(comb_t, dim=0, split_size_or_sections=1)
        return {"content": lq,
                "style": hq,
                "z": z,
                "dset": path_lq.split('/')[-3],
                "n": int(path_lq.split('_')[-1].split('.')[0])}


class BPAEC3D(BPAECDataset):
    def __init__(self, data_glob, transform, resize_size=None, pad=None):
        super().__init__(data_glob=data_glob, transform=transform, resize_size=resize_size)
        self.pad = pad
        self.groups = [list(sorted(v)) for _, v in itertools.groupby(self.file_names, key=group_fn)]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        imgs = self.groups[idx]
        imgs = [cv2.imread(path, 0) for path in imgs]
        if self.resize_size is not None:
            imgs = [cv2.resize(img, dsize=(self.resize_size, self.resize_size)) for img in imgs]
        content = np.stack(imgs, axis=2)
        content = self.transform(content)
        if self.pad is not None:
            content = F.pad(content.permute(1, 2, 0), [self.pad, self.pad], mode="reflect").permute(2, 0, 1)
        d = content.size(0)
        style = content[[(d - 1) // 2]].expand(d, -1, -1)
        return {"content": content.unsqueeze(1),
                "style": style.unsqueeze(1)}
