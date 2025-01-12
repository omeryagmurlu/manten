import torch.nn.functional as F


def binary_cross_entropy_with_logits_with_hinge_domain(inp, target, *args, **kwargs):
    """
    This function is a wrapper around F.binary_cross_entropy_with_logits that
    maps the target domain from [-1, 1] to [0, 1] before calling the function.
    """
    assert (target == -1).logical_or(target == 1).all(), "Target must be in {-1, 1}"
    target = (target + 1) / 2
    return F.binary_cross_entropy_with_logits(inp, target, *args, **kwargs)
