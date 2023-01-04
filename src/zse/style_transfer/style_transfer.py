from typing import Type, Union, List, Callable, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
from zse.models.components.loss import style_loss, gram_matrix, content_loss
from zse.models.components.style_transfer_net import StyleTransferNet
from zse.utils.util import imshow

content_layers_default = ['relu3_2']
style_layers_default = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']


def style_transfer(
        cnn: nn.Module,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        content_layers: Optional[List[str]] = None,
        style_layers: Optional[List[str]] = None,
        content_weights: Optional[Union[List[float], float]] = 1.,
        style_weights: Optional[Union[List[float], float]] = 1e5,
        init_random: Optional[bool] = False,
        normalization: Optional[transforms.Normalize] = None,
        optimizer: Optional[Type[Union[optim.Adam, optim.LBFGS]]] = optim.Adam,
        loss_fn: Optional[Callable[..., torch.Tensor]] = functional.l1_loss,
        num_iter: Optional[int] = 300,
        device: Optional[Union[torch.device, str]] = 'cpu',
        verbose: Optional[bool] = False
) -> torch.Tensor:
    """
    Run style transfer for greyscale images!

    Parameters
    ----------
    cnn : torch.nn.Module
        CNN to use for style transfer.
    content_img : torch.Tensor
        Content images as PyTorch Tensor of shape (N, C, H, W).
    style_img : torch.Tensor
        Style images as PyTorch Tensor of shape (N, C, H, W).
    content_layers : List[str], optional
        Layers to use for content loss. The default is ['relu3_2'].
    style_layers : List[str], optional
        Layers to use for style loss. The default is ['relu1_1', 'relu2_1',
        'relu3_1', 'relu4_1', 'relu5_1'].
    content_weights : List[float] or float, optional
        Weights to use for each layer in layers. The default is 1.;
        if List[float] is provided it must have the same length as layers,
        if float is provided this weight is used for all layers.
    style_weights : List[float] or float, optional
        Weights to use for each layer in style_layers. The default is 1e5;
        if List[float] is provided it must have the same length as layers,
        if float is provided this weight is used for all layers.
    init_random : bool, optional
        If this is set to True, initialize the starting image to uniform random noise.
        The default is False.
    normalization : torchvision.transforms.Normalize, optional
        Normalization used when training the CNN.
    optimizer : torch.optim.Adam or torch.optim.LBFGS, optional
        Optimizer to use for the style transfer. The default is torch.optim.Adam.
    loss_fn : Callable[..., torch.Tensor], optional
        Function used to calculate content and style losses. The default is torch.functional.l1_loss.
    num_iter : int, optional
        Number of iterations to update the image.
    device : torch.device or string, optional
        Device to use. The default is 'cpu'.
    verbose : bool, optional
        If this is set to True, periodically show the current style transfer result. The default is False.

    Returns
    -------
    output : torch.Tensor
        Style transfer result with shape (N, 1, H, W)
    """
    if content_layers is None:
        content_layers = content_layers_default
    if style_layers is None:
        style_layers = style_layers_default
    if type(content_weights) == float:
        content_weights = [content_weights] * len(content_layers)
    if type(style_weights) == float:
        style_weights = [style_weights] * len(style_layers)

    if not (isinstance(content_img, torch.Tensor)):
        raise TypeError(f"content_img should be Tensor. Got {type(content_img)}")
    if not (isinstance(style_img, torch.Tensor)):
        raise TypeError(f"style_img should be Tensor. Got {type(style_img)}")
    assert content_img.ndim == 4, f"content_img should be Tensor with 4 dimension. Got {content_img.ndim}"
    assert style_img.ndim == 4, f"style_img should be Tensor with 4 dimension. Got {style_img.ndim}"
    if style_img.size(0) > 1:
        assert content_img.size(0) == style_img.size(0), \
            f"style_img should have batch size 1 or {content_img.size(0)}. Got {style_img.size(0)}"
    assert len(content_layers) == len(content_weights), \
        f"Length of content_weights is {len(content_weights)} but must be the same as the " \
        f"length of content_layers {len(content_layers)}."
    assert len(style_layers) == len(style_weights), \
        f"Length of style_weights is {len(style_weights)} but must be the same as the " \
        f"length of style_layers {len(style_layers)}."

    if verbose:
        plt.figure(dpi=100)
        plt.subplot(1, 2, 1)
        imshow(content_img.detach(), title='Content Source Img.')
        plt.subplot(1, 2, 2)
        imshow(style_img.detach(), title='Style Source Img.')
        plt.tight_layout()
        plt.show()

    model = StyleTransferNet(cnn=cnn, normalization=normalization, layers=content_layers + style_layers)
    model = model.to(device).requires_grad_(False)

    content_img = content_img.to(device)
    content_features = model(content_img)
    content_targets = {cl: content_features[cl].detach() for cl in content_layers}

    style_img = style_img.to(device)
    style_features = model(style_img)
    style_targets = {sl: gram_matrix(style_features[sl]).detach() for sl in style_layers}

    # Initialize output image to content image or noise
    if init_random:
        input_img = torch.rand_like(content_img, device=device)
    else:
        input_img = content_img.clone()

    input_img.requires_grad_(True)

    if optimizer == optim.Adam:
        optimizer = optimizer([input_img], lr=0.05)
    else:
        optimizer = optimizer([input_img])

    if verbose:
        pbar = tqdm(total=num_iter, desc="Style transfer")
    step = [0]
    while step[0] < num_iter:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            features = model(input_img)
            c_loss = content_loss(features, content_targets, content_layers, content_weights, loss_fn)
            s_loss = style_loss(features, style_targets, style_layers, style_weights, loss_fn, target_gram=False)
            loss = c_loss + s_loss
            loss.backward()

            if verbose:
                pbar.set_postfix(content_loss=c_loss.item(), style_loss=s_loss.item())
                pbar.update()

            step[0] += 1
            if verbose and step[0] % (num_iter // 5) == 0:
                imshow(input_img.detach(), f"iteration {step[0]}", dpi=50)
                plt.show()

            return c_loss + s_loss

        if optimizer == optim.Adam:
            closure()
            optimizer.step()
        else:
            optimizer.step(closure)
    if verbose:
        pbar.close()

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img.detach().cpu()
