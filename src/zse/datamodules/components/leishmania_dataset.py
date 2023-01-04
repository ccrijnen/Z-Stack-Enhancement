import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class LeishmaniaDataset(Dataset):
    def __init__(self, data_glob, transform, imsize=None, scale=None):
        assert not (imsize is not None and scale is not None)
        self.file_names = sorted(glob.glob(data_glob))
        self.transform = transform
        self.imsize = imsize
        self.scale = scale

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path_lq = self.file_names[idx]
        path_hq = path_lq.replace("blurred", "clear")
        lq = cv2.imread(path_lq)
        hq = cv2.imread(path_hq)

        if self.imsize is not None:
            lq = cv2.resize(lq, dsize=(self.imsize, self.imsize))
            hq = cv2.resize(hq, dsize=(self.imsize, self.imsize))
        elif self.scale is not None:
            lq = cv2.resize(lq, dsize=None, fx=self.scale, fy=self.scale)
            hq = cv2.resize(hq, dsize=None, fx=self.scale, fy=self.scale)
        comb = np.concatenate([lq, hq], axis=2)
        comb_t = self.transform(comb)
        lq, hq = torch.split(comb_t, dim=0, split_size_or_sections=3)
        return {"content": lq,
                "style": hq,
                "path": path_hq}
