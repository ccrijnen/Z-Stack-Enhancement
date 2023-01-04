import glob
import os

import h5py
import torchvision.transforms as T
from torch.utils.data import Dataset

to_tensor = T.ToTensor()


def read_hdf5(fname):
    dset = h5py.File(fname, 'r')["/pyramid/00"][()]
    tensor = to_tensor(dset).float()
    return tensor


def save_hdf5(fname, data):
    data = data.mul(255).byte()
    with h5py.File(fname, "w") as f:
        pyramid = f.create_group("pyramid")
        pyramid.attrs["spacing"] = 2.0
        pyramid.attrs["scales"] = [1.0, ]
        pyramid.create_dataset(name="00", data=data)


def get_path_deblurring(path, destination):
    split = path.split("/")[-5:]
    split[2] = split[2] + "_deblurring"
    split = [destination] + split
    folder = "/".join(split[:-1])
    os.makedirs(folder, exist_ok=True)
    fname = split[-1]
    return f"{folder}/{fname}"


class ZStackVolDataset(Dataset):
    def __init__(self, data_glob, destination):
        file_names = sorted(glob.glob(data_glob))
        coronal = [f for f in file_names if "coronal" in f]
        v1 = [f for f in coronal if "vol1" in f]
        v2 = [f for f in coronal if "vol2" in f]
        self.style1 = read_hdf5(v1[-3])
        self.style2 = read_hdf5(v2[(len(v2)-1) // 2])
        self.file_names = file_names
        self.destination = destination

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = self.file_names[idx]
        content = read_hdf5(path)
        dest = get_path_deblurring(path, self.destination)
        style = self.style1 if "vol1" in path else self.style2
        return {"content": content,
                "style": style,
                "dest": dest}
