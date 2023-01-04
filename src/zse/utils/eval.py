import glob
import os
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.filters.rank import entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import models, transforms
from zse.models.adain_module import AdaINLitModule2D
from zse.models.components.adain_net import AdaINNet
from zse.models.components.adain_unet import AdaINUNet
from zse.models.components.unet import UNet

warnings.filterwarnings(action='ignore', category=FutureWarning)


def entropy_map(img, threshold, disk_size):
    img = img.squeeze().permute(1, 2, 0).mul(255).byte().detach().cpu().numpy()
    style = rgb2gray(img)
    style = img_as_ubyte(gaussian(style, sigma=1, channel_axis=2, preserve_range=True))
    entropy_img = entropy(style, disk(disk_size))
    entropy_img /= entropy_img.max()
    return torch.Tensor(entropy_img > threshold)


class ImgDataset(Dataset):
    def __init__(self, data_glob):
        self.file_names = sorted(glob.glob(data_glob))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = self.file_names[idx]
        img = cv2.imread(path)
        img = self.transform(img)
        return {"img": img,
                "path": path}


class QuantEval:
    def __init__(self, file_name: str, dset, out_channels, device, num_workers, entropy_thresh=0.0, disk_size=5):
        super().__init__()
        assert 0.0 < entropy_thresh < 1.0

        self.file_name = file_name
        self.loader = DataLoader(dset, batch_size=1, num_workers=num_workers)
        self.device = device
        self.valid_models = ["data", "gatys", "comi", "adain_unet", "adain", "unet"]
        self.entropy_thresh = entropy_thresh
        self.disk_size = disk_size

        vgg19 = models.vgg19(pretrained=True)
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model_to_net = {
            "adain_unet": AdaINUNet(vgg19, norm, out_channels=out_channels),
            "adain": AdaINNet(vgg19, norm, out_channels=out_channels),
            "unet": UNet(vgg19, norm, in_channels=out_channels, out_channels=out_channels),
        }

        self.fid = FrechetInceptionDistance(reset_real_features=False).to(device)
        self.init_fid_real()

        if os.path.exists(file_name):
            self.results = pd.read_excel(file_name)
        else:
            path = os.path.dirname(file_name)
            os.makedirs(path, exist_ok=True)
            self.results = pd.DataFrame(columns=["name", "FID", "PSNR", "SSIM", "PCC", "model", "weights_or_img_dir"])
            self.add_model("data", "Blurry Images", "blurry images")
            self.save_results()

    def init_fid_real(self):
        for batch in self.loader:
            target = batch["style"].to(self.device)
            if target.ndim == 5:
                target = target.squeeze(0)
            if target.size(1) == 1:
                target = target.expand(-1, 3, -1, -1)
            self.fid.update(target.mul(255).byte().detach(), real=True)

    def save_results(self):
        self.results.to_excel(self.file_name, index=False)

    def calculate_metrics(self, generated: torch.Tensor, target: torch.Tensor):
        if generated.ndim == 5:
            generated = generated.squeeze(0)
        if target.ndim == 5:
            target = target.squeeze(0)

        if generated.size(1) == 1:
            self.fid.update(generated.expand(-1, 3, -1, -1).mul(255).byte(), real=False)
        else:
            self.fid.update(generated.mul(255).byte(), real=False)

        generated = generated[0].permute(1, 2, 0).detach().cpu().numpy()
        target = target[0].permute(1, 2, 0).detach().cpu().numpy()

        psnr = 0.0
        for j in range(generated.shape[2]):
            psnr += peak_signal_noise_ratio(target[:, :, j], generated[:, :, j], data_range=1)
        psnr /= generated.shape[2]
        ssim = structural_similarity(target, generated, data_range=1, multichannel=True)
        pcc = pearsonr(target.flatten(), generated.flatten())[0]
        return psnr, ssim, pcc

    def add_model(self, model, name, weights_or_img_dir):
        assert model in self.valid_models

        if (self.results["name"] == name).any():
            print(f"Already calculated '{name}': '{model}'!")
            return
        print(f"Adding '{name}': '{model}' to results.")

        if model == "data":
            generator = None
        elif model in ["gatys", "comi"]:
            dset = ImgDataset(weights_or_img_dir)
            generator = DataLoader(dset, batch_size=1)
            assert len(generator) == len(self.loader)
        else:
            generator = AdaINLitModule2D.\
                load_from_checkpoint(weights_or_img_dir, strict=False, net=self.model_to_net[model]).to(self.device)
            generator.eval()
            generator.freeze()

        psnr = []
        ssim = []
        pcc = []

        if model not in ["gatys", "comi"]:
            iterator = self.loader
        else:
            iterator = zip(self.loader, generator)

        for data in iterator:
            if model == "data":
                style = data["style"].to(self.device)
                out = data["content"].to(self.device)
            elif model in ["gatys", "comi"]:
                batch: dict = data[0]
                outputs: dict = data[1]
                assert os.path.basename(batch["path"][0]) == os.path.basename(outputs["path"][0])
                style = batch["style"].to(self.device)
                out = outputs["img"].to(self.device)
            else:
                style = data["style"].to(self.device)
                content = data["content"].to(self.device)
                out = generator({"content": content, "style": style})

            if self.entropy_thresh > 0.0:
                mask = entropy_map(style, self.entropy_thresh, self.disk_size).to(self.device)
                style *= mask
                out *= mask

            psnr_i, ssim_i, pcc_i = self.calculate_metrics(out, style)
            psnr.append(psnr_i)
            ssim.append(ssim_i)
            pcc.append(pcc_i)

        self.results.loc[len(self.results)] = [name, self.fid.compute().item(), np.mean(psnr),
                                               np.mean(ssim), np.mean(pcc), model, weights_or_img_dir]
        self.save_results()
        self.fid.reset()
