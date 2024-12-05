from tqdm import tqdm
from accelerate.state import PartialState

state = PartialState()


def progbar(*args, desc="", **kwargs):
    pbar = tqdm(*args, disable=not state.is_local_main_process, **kwargs)
    pbar.set_description(desc)
    return pbar
