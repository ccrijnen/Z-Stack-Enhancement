from abc import ABC
from typing import Any, List, Union

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, PearsonCorrCoef
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid
from zse.models.components.unet import UNet
from zse.models.components.adain_net import AdaINNet
from zse.models.components.adain_unet import AdaINUNet


class AdaINLitModule(LightningModule, ABC):
    def __init__(
            self,
            net: Union[AdaINUNet, AdaINNet, UNet],
            lr: float = 1e-4,
            weight_decay: float = 0,
            lr_decay: float = 0.99,
            style_weight: float = 1e1,
    ):
        super().__init__()
        self.net = net

        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.train_pcc = PearsonCorrCoef(data_range=1)

        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.val_pcc = PearsonCorrCoef(data_range=1)
        self.val_fid = FrechetInceptionDistance(feature=2048, reset_real_features=False)

        self.test_psnr = PeakSignalNoiseRatio(data_range=1)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.test_pcc = PearsonCorrCoef(data_range=1)
        self.test_fid = FrechetInceptionDistance(feature=2048, reset_real_features=False)

    def on_epoch_end(self):
        self.train_psnr.reset()
        self.train_ssim.reset()
        self.train_pcc.reset()

        self.val_psnr.reset()
        self.val_ssim.reset()
        self.val_pcc.reset()
        self.val_fid.reset()

        self.test_psnr.reset()
        self.test_ssim.reset()
        self.test_pcc.reset()
        self.test_fid.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.net.decoder.parameters(), lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=self.hparams.lr_decay)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.lr_decay)
        # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (1.0 + self.hparams.lr_decay * step))
        # [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]
        return [optimizer], [lr_scheduler]

    def forward(self, batch, alpha=1.0):
        content, style = batch["content"], batch["style"]
        if isinstance(self.net, AdaINUNet) or isinstance(self.net, AdaINNet):
            return self.net(content, style, alpha)
        else:
            return self.net(content, style)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self(batch).detach()

    def training_step(self, batch: Any, batch_idx: int):
        out, content_loss, style_loss = self.step(batch)
        content, style = batch["content"], batch["style"]

        style_loss = self.hparams.style_weight * style_loss
        loss = content_loss + style_loss

        psnr = self.train_psnr(out, style).item()
        ssim = self.train_ssim(out, style).item()
        pcc = 0
        for i in range(out.size(0)):
            pcc += self.train_pcc(out[i].view(-1), style[i].view(-1)).item()
        pcc = pcc / out.size(0)

        self.log("train/loss/total", loss.item(),           on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss/content", content_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss/style", style_loss.item(),     on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/psnr", psnr,                        on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ssim", ssim,                        on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/pcc", pcc,                          on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        out, content_loss, style_loss = self.step(batch)
        content, style = batch["content"], batch["style"]

        if out.size(1) == 1:
            self.val_fid.update(out.expand(-1, 3, -1, -1).mul(255).byte().detach(), real=False)
        else:
            self.val_fid.update(out.mul(255).byte().detach(), real=False)

        if self.current_epoch == 0:
            if style.size(1) == 1:
                self.val_fid.update(style.expand(-1, 3, -1, -1).mul(255).byte().detach(), real=True)
            else:
                self.val_fid.update(style.mul(255).byte().detach(), real=True)

        style_loss = self.hparams.style_weight * style_loss
        loss = content_loss + style_loss

        psnr = self.val_psnr(out, style).item()
        ssim = self.val_ssim(out, style).item()
        pcc = 0
        for i in range(out.size(0)):
            pcc += self.val_pcc(out[i].view(-1), style[i].view(-1)).item()
        pcc = pcc / out.size(0)

        self.log("val/loss/total", loss.item(),           on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/loss/content", content_loss.item(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/loss/style", style_loss.item(),     on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/psnr", psnr,                        on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ssim", ssim,                        on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/pcc", pcc,                          on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss.item(), "out": out.detach(), "content": content.detach(), "style": style.detach()}

    def validation_epoch_end(self, outputs: List[Any]):
        out = outputs[0]['out']
        content = outputs[0]['content']
        style = outputs[0]['style']
        stack = torch.stack([content[:32], style[:32], out[:32]], dim=1)
        grid = make_grid(stack.view(-1, *stack.shape[2:]), nrow=3, value_range=(0, 1)).detach()
        self.logger.experiment.add_image(f'val-content-style-out', grid, self.current_epoch, dataformats="CHW")
        fid = self.val_fid.compute().detach()
        self.log("val/fid", fid, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        out, content_loss, style_loss = self.step(batch)
        content, style = batch["content"], batch["style"]

        if out.size(1) == 1:
            self.test_fid.update(out.expand(-1, 3, -1, -1).mul(255).byte(), real=False)
            self.test_fid.update(style.expand(-1, 3, -1, -1).mul(255).byte(), real=True)
        else:
            self.test_fid.update(out.mul(255).byte(), real=False)
            self.test_fid.update(style.mul(255).byte(), real=True)

        style_loss = self.hparams.style_weight * style_loss
        loss = content_loss + style_loss

        psnr = self.test_psnr(out, style).item()
        ssim = self.test_ssim(out, style).item()
        pcc = 0
        for i in range(out.size(0)):
            pcc += self.test_pcc(out[i].view(-1), style[i].view(-1)).item()
        pcc = pcc / out.size(0)

        self.log("test/loss/total", loss.item(),           on_step=False, on_epoch=True)
        self.log("test/loss/content", content_loss.item(), on_step=False, on_epoch=True)
        self.log("test/loss/style", style_loss.item(),     on_step=False, on_epoch=True)
        self.log("test/psnr", psnr,                        on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/ssim", ssim,                        on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/pcc", pcc,                          on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss.item()}

    def test_epoch_end(self, outputs: List[Any]):
        fid = self.test_fid.compute().detach()
        self.log("test/fid", fid, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


class AdaINLitModule2D(AdaINLitModule, ABC):
    def __init__(
            self,
            net: AdaINUNet,
            lr: float = 1e-4,
            weight_decay: float = 0,
            lr_decay: float = 0.99,
            style_weight: float = 1e1,
    ):
        super().__init__(net=net, lr=lr, weight_decay=weight_decay, lr_decay=lr_decay, style_weight=style_weight)
        self.save_hyperparameters(logger=False, ignore=['net'])

    def step(self, batch: Any):
        content, style = batch["content"], batch["style"]
        out, content_loss, style_loss = self.net.loss_2d(content, style)
        return out, content_loss, style_loss


class AdaINLitModule3D(AdaINLitModule, ABC):
    def __init__(
            self,
            net: AdaINUNet,
            lr: float = 1e-4,
            weight_decay: float = 0,
            lr_decay: float = 0.99,
            style_weight: float = 1e1,
            small_style: bool = True,
    ):
        super().__init__(net=net, lr=lr, weight_decay=weight_decay, lr_decay=lr_decay, style_weight=style_weight)
        self.save_hyperparameters(logger=False, ignore=['net'])

    def step(self, batch: Any):
        content, style = batch["content"], batch["style"]
        out, content_loss, style_loss = self.net.loss_3d(content, style, self.hparams.small_style)
        batch["content"] = content.view(-1, *content.shape[-3:])
        batch["style"] = style.view(-1, *style.shape[-3:])
        return out.view(-1, *out.shape[-3:]), content_loss, style_loss
