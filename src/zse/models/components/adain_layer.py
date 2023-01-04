import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mu_x = x.mean(dim=[2, 3], keepdim=True)
        mu_y = y.mean(dim=[2, 3], keepdim=True)

        sigma_x = x.std(dim=[2, 3], keepdim=True) + self.eps
        sigma_y = y.std(dim=[2, 3], keepdim=True) + self.eps

        return (x - mu_x) / sigma_x * sigma_y + mu_y
