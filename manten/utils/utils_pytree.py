from functools import wraps

import optree


def with_tree_map(map_fn):
    def decorator(wrappee):
        @wraps(wrappee)
        def wrapper(*args, **kwargs):
            return optree.tree_map(map_fn, wrappee(*args, **kwargs))

        return wrapper

    return decorator
