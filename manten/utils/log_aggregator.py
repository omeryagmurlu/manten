from enum import Enum

import einops
import numpy as np
from accelerate.utils import gather_object

from manten.metrics.base_metric import BaseMetric
from manten.utils.utils_decorators import with_name_resolution


def argmedian(x):
    return np.argpartition(x, len(x) // 2)[len(x) // 2]


@with_name_resolution
class Reduction(Enum):
    MEAN = "mean"
    SUM = "sum"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"
    LAST = -1
    FIRST = 0

    def reduce(self, arr):
        match self:
            case Reduction.MEAN | Reduction.SUM | Reduction.MAX | Reduction.MIN:
                return einops.reduce(arr, "b ... -> ...", reduction=self.value)
            case Reduction.MEDIAN:
                # return np.median(arr, axis=0) # this doesn't work, [1, 2] -> 1.5 not in [1, 2]
                return arr[argmedian(arr)]
            case Reduction.LAST:
                return arr[-1]
            case Reduction.FIRST:
                return arr[0]
            case _:
                raise ValueError(f"Unknown reduction: {self}")

    def index(self, arr):
        match self:
            case Reduction.MEAN:
                raise ValueError("MEAN does not have an index")
            case Reduction.SUM:
                raise ValueError("SUM does not have an index")
            case Reduction.MEDIAN:
                return argmedian(arr)
            case Reduction.MAX:
                return np.argmax(arr)
            case Reduction.MIN:
                return np.argmin(arr)
            case Reduction.LAST:
                return len(arr) - 1
            case Reduction.FIRST:
                return 0
            case _:
                raise ValueError(f"Unknown reduction: {self}")

    def supports_index(self):
        return self in {
            Reduction.MEDIAN,
            Reduction.MAX,
            Reduction.MIN,
            Reduction.LAST,
            Reduction.FIRST,
        }


class LogAggregator:
    def __init__(self, reductions: list[Reduction] | None = None):
        if reductions is None:
            reductions = [Reduction.MEAN, Reduction.MIN, Reduction.MAX]
        self.reductions: list[Reduction] = [Reduction.resolve(r) for r in reductions]
        self.prefix = ""

        self.list_of_metrics: list[BaseMetric] = []
        self.list_of_dicts = []
        self.did_all_gather = False

    def reset(self):
        self.list_of_metrics = []
        self.list_of_dicts = []
        self.did_all_gather = False

    def log(self, metric: BaseMetric, copy=True):
        if copy:
            metric = metric.copy()
        self.list_of_metrics.append(metric)

    def collate(self, prefix=None, reset=True):
        if not self.list_of_metrics:
            retval = {}
        else:
            # TODO: This assumes that the keys are the same for all logs
            self.list_of_dicts = [metric.metrics() for metric in self.list_of_metrics]
            keys = self.list_of_dicts[0].keys()

            retval = {}
            nparr = {}  # this only exists so that we can have a usable ordering for the metrics
            for key in keys:
                nparr[key] = [logs[key] for logs in self.list_of_dicts]
            for reduction in self.reductions:
                for key in keys:
                    retval[f"{reduction.name.lower()}/{key}"] = reduction.reduce(nparr[key])

        if reset:
            self.reset()
        return self._rename_logs(retval, prefix)

    def log_collate(self, logs, *args, **kwargs):
        if len(self.list_of_metrics) > 0:
            raise ValueError("log_collate must be called after collate or log_collate")
        self.log(logs, copy=False)
        return self.collate(*args, **kwargs)

    def all_gather(self):
        if self.did_all_gather:
            raise ValueError("all_gather can only be called once")
        if not self.list_of_metrics:
            return
        metric_proto = self.list_of_metrics[0]
        l_of_states = [metric.state_dict() for metric in self.list_of_metrics]

        gathered_states = gather_object(l_of_states)

        l_of_metrics = []
        for state in gathered_states:
            metric = metric_proto.copy()
            metric.load_state_dict(state)
            l_of_metrics.append(metric)
        self.list_of_metrics = l_of_metrics

    def create_vis_logs(self, sort_key):
        valid_reductions = [r for r in self.reductions if r.supports_index()]

        list_of_sortkey_values = [logs[sort_key] for logs in self.list_of_dicts]

        retval = {}
        for reduction in valid_reductions:
            index = reduction.index(list_of_sortkey_values)
            metric = self.list_of_metrics[index]
            visualisation_logs = metric.visualize(
                sort_key=sort_key, reduction_hint=reduction.name.lower()
            )
            if visualisation_logs is not None:
                for vis_key, vis in visualisation_logs.items():
                    retval[f"{reduction.name.lower()}:{sort_key}/{vis_key}"] = vis

        return self._rename_logs(retval)

    def _rename_logs(self, logs, prefix=None):
        if prefix is not None:
            self.prefix = prefix

        if not logs:
            return {}

        return {f"{self.prefix}{key}": value for key, value in logs.items()}
