
# code taken and amended from # Shariatnia, M. M. (2021). Simple CLIP (Version 1.0.0) [Computer software]. https://doi.org/10.5281/zenodo.6845731

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]