import argparse
import os
from abc import ABC

import h5py
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader
from torchvision import models, transforms
from zse.datamodules.components.brain_datasets import ZStackDataset3D
from zse.models.adain_module import AdaINLitModule2D
from zse.models.components.adain_unet import AdaINUNet
from zse.models.components.unet import UNet

home = "/p/fastdata/bigbrains/personal/crijnen1"
data_root = f"{home}/data"
zse_path = f"{home}/Z-Stack-Enhancement"
exp_path = f"{zse_path}/logs/experiments/runs/brain_1micron"

vgg19 = models.vgg19(pretrained=True)
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model_to_net = {
    "adain_unet": AdaINUNet(vgg19, norm),
    "unet": UNet(vgg19, norm),
}
checkpoints = {
    "adain_unet_3ds": f"{exp_path}/3d/unet/gram_loss/lr:0.001-style_weight:1000000.0-weight_decay:0-lr_decay:1-"
                      f"small_style:True/2022-11-29_18-01-24/checkpoints/epoch_016.ckpt",
    "adain_unet_3d": f"{exp_path}/3d/unet/gram_loss/lr:0.001-style_weight:1000000.0-weight_decay:0-lr_decay:1-"
                     f"small_style:False/2022-11-29_18-07-15/checkpoints/epoch_019.ckpt",
    "adain_unet_2d": f"{exp_path}/2d/unet/gram_loss/lr:0.001-style_weight:1000000.0-weight_decay:0-lr_decay:1"
                     f"/2022-11-29_17-55-04/checkpoints/epoch_018.ckpt",
    "unet_3ds": f"{exp_path}/3d/u17/adain_loss/lr:0.0001-style_weight:100.0-weight_decay:0-lr_decay:1-"
                f"small_style:True/2022-11-29_17-57-54/checkpoints/epoch_016.ckpt",
}


def save_hdf5_batch(output_dir, paths, data):
    assert data.ndim == 5
    data = data.mul(255).byte().squeeze(2).permute(0, 2, 3, 1).detach().cpu()
    for i, path in enumerate(paths):
        name = path.split("/")[-1]
        file = h5py.File(f"{output_dir}/{name}", "w")
        file.create_dataset("z_stack", data[i].shape, dtype="u1", data=data[i])
        file.close()


class CustomWriter(BasePredictionWriter, ABC):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        save_hdf5_batch(self.output_dir, batch["path"], prediction)


def save_preds(model_type):
    output_dir = f"{data_root}/bigbrain_1micron/20/test/{model_type}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    model_name = "adain_unet" if "adain_unet" in model_type else "unet"
    module = AdaINLitModule2D.load_from_checkpoint(checkpoints[model_type], net=model_to_net[model_name])
    module.freeze()
    module.eval()

    data_test = ZStackDataset3D(f"{data_root}/bigbrain_1micron/20/test/blurry/*.hdf5", transform=transforms.ToTensor())
    loader = DataLoader(dataset=data_test, batch_size=1, num_workers=32, pin_memory=True, shuffle=False)

    pred_writer = CustomWriter(output_dir=output_dir, write_interval="batch")
    trainer = Trainer(accelerator="gpu", strategy="ddp", devices=4, callbacks=[pred_writer])
    trainer.predict(module, loader, return_predictions=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='type of model')

    args = parser.parse_args()
    model = args.model

    save_preds(model)
