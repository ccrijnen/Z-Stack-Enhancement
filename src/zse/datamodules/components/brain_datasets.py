import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from zse.utils.data_utils import resample, read_h5


class ZStackDataset2D(Dataset):
    def __init__(self, data_glob, transform, resample=False, z_target=29, z_depth=29, binary=False):
        assert z_target <= z_depth
        if resample:
            z_target = z_depth = 20
        start_idx = 0
        if z_target < z_depth:
            start_idx = (z_depth - z_target + 1) // 2
        if binary:
            z_target += 1

        file_names = sorted(glob.glob(data_glob))
        indices = np.tile(np.arange(start_idx, start_idx+z_target), len(file_names))
        file_names = np.repeat(file_names, z_target)

        self.file_names = list(zip(file_names, indices))
        self.transform = transform
        self.z_depth = z_depth
        self.z_target = z_target - 1
        self.binary = binary
        self.resample = resample

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path, depth = self.file_names[idx]
        z_stack = read_h5(path)
        if self.resample:
            z_stack = resample(z_stack)
        z_stack = self.transform(z_stack)
        style = z_stack[(self.z_depth - 1) // 2]
        if self.binary and depth == self.z_target:
            content = (style > 0.4).float()
        else:
            content = z_stack[depth]
        return {"content": content.unsqueeze(0), "style": style.unsqueeze(0), "path": path, "z": depth}


class ZStackDataset3D(Dataset):
    def __init__(self, data_glob, transform, resample=False, z_target=29):
        self.file_names = sorted(glob.glob(data_glob))
        self.transform = transform
        self.resample = resample
        self.z_target = z_target

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = self.file_names[idx]
        z_stack: np.ndarray = read_h5(path)
        z = z_stack.shape[2]
        start_idx = 0
        if self.z_target < z:
            start_idx = (z - self.z_target + 1) // 2
        z_stack = z_stack[:, :, start_idx:start_idx+self.z_target]
        if self.resample:
            z_stack = resample(z_stack)
        z_stack: torch.Tensor = self.transform(z_stack)
        d = z_stack.size(0)
        style = z_stack[[(d-1) // 2]].expand(d, -1, -1)
        return {"content": z_stack.unsqueeze(1), "style": style.unsqueeze(1), "path": path}
