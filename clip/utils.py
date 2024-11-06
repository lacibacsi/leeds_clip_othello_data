
import torch

from othello_game import othello


def get_device():
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return torch.device(device)


def get_lr(optimizer):
    # code taken and amended from # Shariatnia, M. M. (2021). Simple CLIP (Version 1.0.0) [Computer software]. https://doi.org/10.5281/zenodo.6845731
    for param_group in optimizer.param_groups:
        return param_group["lr"]

