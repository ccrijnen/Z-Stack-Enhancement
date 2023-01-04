from typing import List, Dict, Callable

import torch


def style_loss_adain(out_features: Dict[str, torch.Tensor],
                     style_features: Dict[str, torch.Tensor],
                     style_layers: List[str],
                     style_weights: List[float],
                     loss_fn: Callable[..., torch.Tensor]):
    """
    Computes the style loss using mean and standard deviation used in the paper
    'Arbitrary style transfer in real-time with adaptive instance normalization'
    at a set of layers.

    Parameters
    ----------
    out_features : Dict[str, torch.Tensor]
        Dictionary containing intermediate features for all content and style layers
        of the current image.
    style_features : Dict[str, torch.Tensor]
        Dictionary containing intermediate features for all content and style layers
        of the target image.
    style_layers : List[str]
        List of layers to include in the content loss. Keys for the features' dictionary.
    style_weights : List[float]
        List of scalars giving the weight for the style loss at the corresponding layer
        in style_layers; must have the same length as style_layers
    loss_fn : Callable[..., torch.Tensor], optional
        Function used to calculate style loss.

    Returns
    -------
    s_loss : torch.Tensor
        A PyTorch Tensor holding a scalar giving the style loss.
    """
    batch_size = next(iter(out_features.values())).size(0)
    s_loss = 0
    if len(style_features) == 0:
        return s_loss
    for s_weight, s_layer in zip(style_weights, style_layers):
        out_f = out_features[s_layer]
        style_f = style_features[s_layer]
        if style_f.size(0) == 1:
            style_f = style_f.expand(batch_size, -1, -1, -1)

        mu_out = out_f.mean(dim=[2, 3])
        mu_style = style_f.mean(dim=[2, 3])
        s_loss += s_weight * loss_fn(mu_out, mu_style)

        sigma_out = out_f.std(dim=[2, 3])
        sigma_style = style_f.std(dim=[2, 3])
        s_loss += s_weight * loss_fn(sigma_out, sigma_style)
    return s_loss


def style_loss(out_features: Dict[str, torch.Tensor],
               target_features: Dict[str, torch.Tensor],
               style_layers: List[str],
               style_weights: List[float],
               loss_fn: Callable[..., torch.Tensor],
               target_gram: bool = True):
    """
    Computes the style loss at a set of layers.

    Parameters
    ----------
    out_features : Dict[str, torch.Tensor]
        Dictionary containing intermediate features for all content and style layers
        of the current image.
    target_features : Dict[str, torch.Tensor]
        Dictionary containing intermediate features for all style layers of the target image.
    style_layers : List[str]
        List of layers to include in the content loss. Keys for the features' dictionary.
    style_weights : List[float]
        List of scalars giving the weight for the style loss at the corresponding layer
        in style_layers; must have the same length as style_layers
    loss_fn : Callable[..., torch.Tensor]
        Function used to calculate style loss.
    target_gram: bool, optional
        If set to true, compute the gram matrices of the target features. Otherwise, assumes that the gram matrices
        have been precomputed.

    Returns
    -------
    loss : torch.Tensor
        A PyTorch Tensor holding a scalar giving the style loss.
    """
    batch_size = next(iter(out_features.values())).size(0)
    loss = 0
    if len(target_features) == 0:
        return loss
    for weight, layer in zip(style_weights, style_layers):
        out_f = out_features[layer]
        target_f = target_features[layer]
        if target_f.size(0) == 1:
            target_f = target_f.expand(batch_size, *[-1]*(target_f.ndim-1))
        if target_gram:
            target_f = gram_matrix(target_f)
        loss += weight * loss_fn(gram_matrix(out_f), target_f)
    return loss


def gram_matrix(x, normalize=True):
    """
    Compute the Gram matrix from a feature Tensor.

    Parameters
    ----------
    x : torch.Tensor
        PyTorch Tensor of shape (N, C, H, W) giving features for a batch of N images.
    normalize : bool, optional
        If this is set to True, divide the Gram matrix by the number of neurons (C * H * W).
        The default is False.

    Returns
    -------
    gram : torch.Tensor
        PyTorch Tensor of shape (N, C, C) giving the (optionally normalized)
        Gram matrices for the N input images.
    """
    n, c, h, w = x.size()
    x = x.view(n, c, h * w)

    gram = x.bmm(x.transpose(1, 2))

    if normalize:
        gram = gram.div(c * h * w)

    return gram


def content_loss(out_features: Dict[str, torch.Tensor],
                 target_features: Dict[str, torch.Tensor],
                 content_layers: List[str],
                 content_weights: List[float],
                 loss_fn: Callable[..., torch.Tensor]):
    """
    Computes the content loss at a set of layers.

    Parameters
    ----------
    out_features : Dict[str, torch.Tensor]
        Intermediate features for all content and style layers of the current image.
    target_features : Dict[str, torch.Tensor]
        Dictionary containing intermediate features for all content layers of the target image.
    content_layers : List[str]
        List of layers to include in the content loss. Keys for the features' dictionary.
    content_weights : List[float]
        List of scalars giving the weight for the content loss at the corresponding layer
        in layers; must have the same length as layers
    loss_fn : Callable[..., torch.Tensor]
        Function used to calculate content loss.

    Returns
    -------
    loss : torch.Tensor
        A PyTorch Tensor holding a scalar giving the content loss.
    """
    loss = 0
    if len(target_features) == 0:
        return loss
    for weight, layer in zip(content_weights, content_layers):
        out_f = out_features[layer]
        target_f = target_features[layer]
        loss += weight * loss_fn(out_f, target_f)
    return loss
