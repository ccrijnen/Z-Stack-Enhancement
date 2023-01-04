from collections.abc import Iterable
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models._utils import IntermediateLayerGetter


def layer_search(obj, layer_list=None):
    if layer_list is None:
        layer_list = []
    if isinstance(obj, nn.Sequential) or isinstance(obj, Iterable):
        for o in obj:
            layer_search(o, layer_list)
    else:
        layer_list.append(obj)
    return layer_list


class StyleTransferNet(nn.Module):
    def __init__(self, cnn: nn.Module,
                 layers: List[str],
                 normalization: Optional[T.Normalize],
                 original: Optional[bool] = False) -> None:
        """
        Creates the style transfer model from the cnn.

        Parameters
        ----------
        cnn : torch.nn.Module
            CNN to use for style transfer.
        normalization : torchvision.transforms.Normalize or None
            Normalization used when training the CNN.
        layers : List[str]
            List of layers to calculate intermediate features for.
        original : bool

        """
        super().__init__()
        model, out_channels = self.build(cnn, layers, original)
        return_layers = dict(zip(layers, layers))
        self.layers = layers
        self.in_channels = model.conv1_1.in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.model = IntermediateLayerGetter(model, return_layers=return_layers)

    @staticmethod
    def build(cnn: nn.Module,
              layers: List[str],
              original: bool = False) -> Tuple[nn.Sequential, List[int]]:
        """
        Build the style transfer model by iterating through the cnn layers and
        renaming the all layers to '{layer name}{block}_{conv # in the block}'

        Parameters
        ----------
        cnn : torch.nn.Module
            CNN to use for style transfer.
        layers : List[str]
            List of layers to calculate intermediate features for.
        original : bool


        Returns
        -------
        model : nn.Sequential
            The style transfer model.
        out_channels : List[int]

        """
        cnn.eval()

        model = nn.Sequential()

        out_channels = []
        cnn_layers = layer_search(cnn.children())
        block, number, last_layer, last_block = 1, 1, 0, 0
        for i, layer in enumerate(cnn_layers):
            if isinstance(layer, nn.Conv2d):
                name = f'conv{block}_{number}'
                # layer = nn.Conv2d(layer.in_channels, layer.out_channels, kernel_size=layer.kernel_size,
                #                   padding=layer.padding, padding_mode="reflect")
                layer.padding_mode = 'reflect'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn{block}_{number}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu{block}_{number}'
                if not original:
                    layer = nn.ReLU(inplace=False)
                number += 1
            elif isinstance(layer, nn.MaxPool2d):
                out_channels.append(model.get_submodule(f"conv{block}_{number-1}").out_channels)
                name = f'pool{block}'
                if not original:
                    layer = nn.AvgPool2d(layer.kernel_size, layer.stride)
                # else:
                #     layer.ceil_mode = True
                block += 1
                number = 1
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in layers:
                last_layer = i
                last_block = block
        return model[:last_layer+2], out_channels[:last_block]

    def forward(self, x):
        if x.size(1) == 1:
            x = x.expand(-1, self.in_channels, -1, -1)
        if self.normalization:
            x = self.normalization(x)
        out = self.model(x)
        return out
