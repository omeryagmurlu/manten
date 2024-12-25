def get_logger(*args, **kwargs):
    try:
        from accelerate.logging import get_logger as _get_logger

        return _get_logger(*args, **kwargs)
    except ImportError:
        from logging import getLogger

        return getLogger(*args, **kwargs)
