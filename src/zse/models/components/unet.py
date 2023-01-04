from collections import OrderedDict
from functools import partial
from typing import List, Optional

import celldetection as cd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from zse.models.components.loss import style_loss_adain, style_loss
from zse.models.components.style_transfer_net import StyleTransferNet


xz = T.Lambda(lambda x: x.transpose(1, 3))
yz = T.Lambda(lambda x: x.transpose(1, 4))

content_layer_default = 'relu4_1'
style_layers_default = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4']


class UNet(nn.Module):
    def __init__(
            self,
            vgg: nn.Module,
            normalization: Optional[T.Normalize] = None,
            content_layer: Optional[str] = None,
            style_layers: Optional[List[str]] = None,
            in_channels: int = 1,
            out_channels: int = 1,
            style_loss_fn: str = "adain"
    ):
        super().__init__()
        assert style_loss_fn in ["adain", "gram"]
        if content_layer is None:
            content_layer = content_layer_default
        if style_layers is None:
            style_layers = style_layers_default

        loss_net = StyleTransferNet(
            cnn=vgg.features,
            normalization=normalization,
            layers=sorted(list(set(style_layers+[content_layer]))),
            original=True
        )
        loss_net = loss_net.requires_grad_(False)

        self.in_channel = in_channels
        self.out_channel = out_channels
        self.content_layer = content_layer
        self.style_layers = style_layers
        self.encoder = loss_net
        self.decoder = cd.models.U17(in_channels, out_channels)
        cd.tweak_module_(self.decoder, nn.Conv2d, padding_mode="reflect")
        self.output_activation = nn.Sigmoid()

        self.loss_fn = nn.L1Loss()
        self.style_loss = partial(style_loss_adain if style_loss_fn == "adain" else style_loss,
                                  style_layers=style_layers,
                                  style_weights=[1.]*len(style_layers),
                                  loss_fn=self.loss_fn)

    def forward(self, content, style):
        assert (content.ndim == 4 and style.ndim == 4) or (content.ndim == 5 and style.ndim == 5)
        assert content.size() == style.size()
        shape = content.size()

        if len(shape) == 5:
            content = content.view(-1, *shape[-3:])

        out = self.decoder(content)
        out = self.output_activation(out)

        if len(shape) == 5:
            out = out.view(*shape)

        return out

    def loss_2d(self, content, style):
        assert content.ndim == 4 and style.ndim == 4

        out = self(content, style)
        content_features = self.encoder(content)
        style_features = self.encoder(style)
        out_features = self.encoder(out)

        content_enc = content_features[self.content_layer]
        out_enc = out_features[self.content_layer]

        content_loss = self.loss_fn(out_enc, content_enc)
        style_loss = self.style_loss(out_features, style_features)

        return out, content_loss, style_loss

    def loss_3d(self, content, style, small_style: bool = True):
        assert content.ndim == 5 and style.ndim == 5

        n, d, c, h, w = content.size()
        style_shrunk = style[:, 0]

        out = self(content, style)
        content_features = self.encoder(content.view(-1, *content.shape[-3:]))
        style_features = self.encoder(style.view(-1, *style.shape[-3:]))
        out_features = self.encoder(out.view(-1, *out.shape[-3:]))

        out_xz = xz(out).reshape(-1, c, d, w)
        out_yz = yz(out).reshape(-1, c, h, d)

        out_features_xz = self.encoder(out_xz)
        out_features_yz = self.encoder(out_yz)

        content_enc = content_features[self.content_layer]
        out_enc = out_features[self.content_layer]

        content_loss = self.loss_fn(out_enc, content_enc)
        style_loss = 0.8 * self.style_loss(out_features, style_features)

        if small_style:
            pad_xz = F.pad(style_shrunk, (0, 0, d // 2, d // 2), mode="reflect").unsqueeze_(1).expand(-1, h, -1, -1, -1)
            pad_yz = F.pad(style_shrunk, (d // 2, d // 2, 0, 0), mode="reflect").unsqueeze_(1).expand(-1, w, -1, -1, -1)

            all_idx_xz = torch.arange(h)[:, None] + torch.arange(d)
            all_idx_yz = torch.arange(w)[:, None] + torch.arange(d)
            style_xz = pad_xz[:, torch.arange(all_idx_xz.shape[0])[:, None], :, all_idx_xz, :]
            style_yz = pad_yz[:, torch.arange(all_idx_yz.shape[0])[:, None], :, :, all_idx_yz]
            style_xz = style_xz.permute(2, 0, 3, 1, 4).reshape(-1, c, d, w)
            style_yz = style_yz.permute(2, 0, 3, 4, 1).reshape(-1, c, h, d)

            style_features_xz = self.encoder(style_xz)
            style_features_yz = self.encoder(style_yz)
            style_loss += 0.1 * self.style_loss(out_features_xz, style_features_xz)
            style_loss += 0.1 * self.style_loss(out_features_yz, style_features_yz)
        else:
            style_features_z = self.encoder(style_shrunk)
            style_features_xz = OrderedDict([(k, v.repeat_interleave(h, dim=0)) for k, v in style_features_z.items()])
            style_features_yz = OrderedDict([(k, v.repeat_interleave(w, dim=0)) for k, v in style_features_z.items()])
            style_loss += 0.1 * self.style_loss(out_features_xz, style_features_xz)
            style_loss += 0.1 * self.style_loss(out_features_yz, style_features_yz)

        return out, content_loss, style_loss
