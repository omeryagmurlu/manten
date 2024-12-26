import einops
import numpy as np
import torch
from mani_skill.utils.wrappers import FrameStack
from optree import tree_flatten, tree_unflatten


class TreeFrameStack(FrameStack):
    @staticmethod
    def __transpose_list_of_list(list_of_list):
        """cause zip and it's tuples..."""
        return [[li[i] for li in list_of_list] for i in range(len(list_of_list[0]))]

    def observation(self, _observation):
        # return torch.stack(list(self.frames)).transpose(0, 1)

        flat_frames = [tree_flatten(f) for f in list(self.frames)]
        obs_pytree_spec = flat_frames[0][1]
        # assumes all obss have the same pytree structure, everything is moot otherwise
        flat_frames = [f[0] for f in flat_frames]
        # list of list of tensors with shape (nhist, #spec, ...)
        elems = self.__transpose_list_of_list(flat_frames)
        # list of list of tensors with shape (#spec, nhist, ...)
        stacked_elems = [
            einops.rearrange(elem, "zero one ... -> one zero ...") for elem in elems
        ]
        stacked_obs = tree_unflatten(obs_pytree_spec, stacked_elems)
        return stacked_obs


class ActionExecutor:
    def execute_action_in_env(self, action, envs):
        return envs.step(action)


class HorizonActionExecutor(ActionExecutor):
    def execute_action_in_env(self, action, envs):
        for i in range(action.shape[1]):
            obs, rew, terminated, truncated, info = envs.step(action[:, i])
            if truncated.any():
                break
        return obs, rew, terminated, truncated, info


class RecedingHorizonActionExecutor(ActionExecutor):
    """Receding horizon action executor that has the option of averaging actions over overlapping horizons.

    Parameters:
    -----------
    receding_horizon : int, optional
        The number of horizons over which to execute actions. If None, it is set to maximum inferred from actions.
    average : str, optional
        The type of averaging to perform. Can be 'cumulative' for non-weighted moving average or 'ema' for exponential moving average.
    ema_alpha : float, optional
        The smoothing factor for exponential moving average.
    maximum_horizon_buf_len : int, optional
        The maximum length of the horizon buffer. As long as it's bigger than ~2 * max_episode_steps, it's fine, you may want to tune this if memory is an issue.

    Methods:
    --------
    execute_action_in_env(act_seq, envs):
        Executes the action sequence in the given environments.

    Example for cumulative averaging:
    ---------------------------------
    If receding_horizon=1, the wrapper will average each action with the act_seq.shape[0] - 1 previous horizons.
    If receding_horizon=act_seq.shape[0] / 2, the wrapper will average actions over act_seq.shape[0]/receding_horizon = 2
    horizons.
    ```
        a|a|a|a|a|a|a|a
        .|.|.|.|a|a|a|a|a|a|a|a
        .|.|.|.|.|.|.|.|a|a|a|a|a|a|a|a
        _______________________
        1|1|1|1|2|2|2|2|2|2|2|2 sequences averaged for action
    ```
    """

    def __init__(
        self,
        *,
        receding_horizon=None,
        average=None,
        ema_alpha=0.75,  # last sample has weight compared to the rest
        maximum_horizon_buf_len=1000,
    ):
        self.__receding_horizon = receding_horizon
        self.__average = average
        self.__ema_alpha = ema_alpha
        self.__maximum_horizon_buf_len = maximum_horizon_buf_len

        self.__reset()

    def __reset(self):
        self.__step = 0
        self.__backend = None
        self.__act_seq = None
        self.__upd_seq = None
        self.__act_horizon = None
        self.__has_setup = False

    def __ifndef_setup(self, act_seq, envs):
        if self.__has_setup:
            return

        num_envs = envs.num_envs

        self.__act_horizon = act_seq.shape[1]

        self.__step = 0
        if isinstance(act_seq, torch.Tensor):
            self.__backend = torch
            self.__act_seq = torch.zeros(
                (num_envs, self.__maximum_horizon_buf_len, envs.action_space.shape[-1]),
                dtype=act_seq.dtype,
                device=act_seq.device,
            )
            self.__upd_seq = torch.zeros(
                (num_envs, self.__maximum_horizon_buf_len, 1),
                dtype=int,
                device=act_seq.device,
            )
        else:
            self.__backend = np
            self.__act_seq = np.zeros(
                (num_envs, self.__maximum_horizon_buf_len, envs.action_space.shape[-1]),
                dtype=act_seq.dtype,
            )
            self.__upd_seq = np.zeros(
                (num_envs, self.__maximum_horizon_buf_len, 1), dtype=int
            )

        self.__has_setup = True

    def execute_action_in_env(self, action, envs):
        self.__ifndef_setup(action, envs)
        self.__update(action)
        assert self.__act_horizon >= self.__receding_horizon

        for _ in range(self.__receding_horizon):
            obs, rew, terminated, truncated, info = envs.step(self.__act_seq[:, self.__step])
            self.__step += 1
            if truncated.any():
                self.__reset()
                break
        return obs, rew, terminated, truncated, info

    def __update(self, act_seq):
        if self.__receding_horizon is None:
            self.__update_direct(act_seq)
            self.__receding_horizon = self.__act_horizon

            return

        if not self.__average:
            self.__update_direct(act_seq)
        elif self.__average == "cumulative":
            self.__update_cumulative(act_seq)
        elif self.__average == "ema":
            self.__update_ema(act_seq)
        else:
            raise ValueError(f"Unknown average type {self.__average}")

    def __update_direct(self, act_seq):
        """Simply overwrite the old actions with the new ones."""
        self.__act_seq[:, self.__step : self.__step + self.__act_horizon] = act_seq

    def __update_cumulative(self, act_seq):
        """Non-weighted moving average. I don't know (and don't want to learn right now)
        how to do FIR wma with python, so here's a simple, non-weighted moving average via cumulative means."""
        acts = self.__act_seq[:, self.__step : self.__step + self.__act_horizon]
        upds = self.__upd_seq[:, self.__step : self.__step + self.__act_horizon]

        # (curr + (prev * (n-1))) / n (cum avg, sum new + n-1 old and divide by n)
        # = curr/n + prev * (n-1)/n
        # = curr/n + prev * (1 - 1/n)
        # = curr/n + prev - prev/n
        # = prev + (curr - prev)/n (add the curr's contribution to the avg)

        self.__act_seq[:, self.__step : self.__step + self.__act_horizon] = acts + (
            act_seq - acts
        ) / (upds + 1)
        self.__upd_seq[:, self.__step : self.__step + self.__act_horizon] = upds + 1

    def __update_ema(self, act_seq):
        """Exponential moving average."""
        acts = self.__act_seq[:, self.__step : self.__step + self.__act_horizon]
        upds = self.__upd_seq[:, self.__step : self.__step + self.__act_horizon]

        # ema@0 = new
        # ema@t = (1 - alpha) * ema@t-1 + alpha * new

        self.__act_seq[:, self.__step : self.__step + self.__act_horizon] = (
            self.__backend.where(
                upds == 0,
                act_seq,  # ema@0
                (1 - self.__ema_alpha) * acts + self.__ema_alpha * act_seq,  # ema@t
            )
        )
        self.__upd_seq[:, self.__step : self.__step + self.__act_horizon] = upds + 1
