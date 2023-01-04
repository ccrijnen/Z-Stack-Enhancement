from typing import Tuple, Optional

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.utils import make_grid


def imshow(tensor: torch.Tensor, title: Optional[str] = None,
           dpi: Optional[int] = None, fname: Optional[str] = None) -> None:
    """
    Plots a greyscale image from a PyTorch Tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Image as a PyTorch Tensor of shape (N, C, H, W).
    title : str, optional
        If this is provided, add a title to the plot.
    dpi : int, optional
        If this is provided, create a new figure with the given dpi.
    fname : str, optional
        If this is provided, save the image with the given fname.
    """
    image = tensor.clone().cpu()
    image = image[0]
    if image.size(0) > 1:
        cmap = None
        image = image.permute(1, 2, 0)
    else:
        cmap = "gray"
        image = image[0]

    if fname is not None:
        plt.imsave(fname, image, cmap=cmap, vmin=0, vmax=1)

    if dpi is not None:
        plt.figure(dpi=dpi)

    plt.imshow(image, cmap=cmap, vmin=0, vmax=1)

    if title is not None:
        plt.title(title)

    plt.axis("off")
    # plt.show()


def grid_show(tensor: torch.Tensor, ncol: Optional[int] = 8, title: Optional[str] = None,
              dpi: Optional[int] = 200, fname: Optional[str] = None) -> None:
    """
    Plots a grid of greyscale images from a PyTorch Tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Image as a PyTorch Tensor of shape (N, C, H, W).
    ncol : int, optional
        Number of columns in the image grid. The default is 8.
    title : str, optional
        If this is provided, add a title to the plot.
    dpi : int, optional
        Dpi of the pyplot figure. The default is (30, 15)
    fname : str, optional
        If this is provided, save the image with the given fname.
    """
    tensor = tensor.clone().cpu()
    grid = make_grid(tensor, nrow=ncol)[0]

    if fname is not None:
        plt.imsave(fname, grid, cmap="gray", vmin=0, vmax=1)

    plt.figure(dpi=dpi)
    plt.imshow(grid, cmap="gray", vmin=0, vmax=1)

    if title is not None:
        plt.title(title, color="white")

    plt.axis("off")
    plt.show()


def plot_cross_section(z_stack: torch.Tensor, idx: Tuple[int, int, int],
                       reverse: Optional[bool] = False, fname: Optional[str] = None) -> None:
    """
    Plots a greyscale cross-section from a 3d z-stack.

    Parameters
    ----------
    z_stack : torch.Tensor
        PyTorch Tensor of shape (1, C, H, W)
    idx : Tuple[int, int, int]
        Cross-section index.
    reverse : bool, optional
        If this is set to True, change all text color and tick markers to white.
    fname : str, optional
        If this is provided, save the image with the given fname.
    """
    color = 'black' if not reverse else 'white'
    z_stack = z_stack.clone().cpu()
    z, y, x = idx
    # yx
    a = z_stack[0, z, :, :]
    # zx = top
    b = z_stack[0, :, y, :]
    # yz = right
    c = z_stack[0, :, :, x].transpose(0, 1)

    fig, main_ax = plt.subplots(figsize=(12, 12))
    divider = make_axes_locatable(main_ax)
    top_ax = divider.append_axes("top", 1.25, pad=0.01, sharex=main_ax)
    right_ax = divider.append_axes("right", 1.25, pad=0.01, sharey=main_ax)

    main_ax.tick_params(colors=color, which='both')
    top_ax.tick_params(colors=color, which='both')
    right_ax.tick_params(colors=color, which='both')

    main_ax.xaxis.set_tick_params(color=color)
    top_ax.xaxis.set_tick_params(labelbottom=False, color=color)
    right_ax.yaxis.set_tick_params(labelleft=False, color=color)

    main_ax.set_xlabel('X', color=color)
    main_ax.set_ylabel('Y', color=color)
    top_ax.set_ylabel('Z', color=color)
    right_ax.set_xlabel('Z', color=color)

    main_ax.imshow(a, cmap='gray', vmin=0, vmax=1)
    top_ax.imshow(b, cmap='gray', vmin=0, vmax=1)
    right_ax.imshow(c, cmap='gray', vmin=0, vmax=1)
    main_ax.axhline(y, color='r')
    main_ax.axvline(x, color='g')
    top_ax.axhline(z, color='r')
    right_ax.axvline(z, color='g')
    main_ax.autoscale(enable=False)
    right_ax.autoscale(enable=False)
    top_ax.autoscale(enable=False)
    right_ax.set_xlim(right=29)
    top_ax.set_ylim(bottom=29)

    if fname is not None:
        plt.savefig(fname)
