# this is needed because we want to load sapien/maniskill only when it's needed,
# and importing ANYTHING from sapien/maniskill will load the whole thing and
# register the envs etc


def wrapper_proxy(name, **outer_kwargs):
    def proxy(*args, **inner_kwargs):
        from . import utils_wrappers

        kwargs = {**outer_kwargs, **inner_kwargs}

        return getattr(utils_wrappers, name)(*args, **kwargs)

    return proxy
