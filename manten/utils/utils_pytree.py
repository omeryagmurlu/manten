import functools

import einops
import optree

from manten.utils.utils_decorators import wraps


def with_tree_map(map_fn):
    def decorator(wrappee):
        @functools.wraps(wrappee)
        def wrapper(*args, **kwargs):
            return optree.tree_map(map_fn, wrappee(*args, **kwargs))

        return wrapper

    return decorator


@wraps(einops.rearrange)
def tree_rearrange(tree, *args, **kwargs):
    return optree.tree_map(lambda tensor: einops.rearrange(tensor, *args, *kwargs), tree)
