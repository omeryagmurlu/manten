from enum import Enum

import einops

from manten.utils.utils_enum import with_name_resolution


@with_name_resolution
class Reduction(Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    LAST = -1
    FIRST = 0


class LogAggregator:
    def __init__(self, reductions: list[Reduction] | None = None):
        if reductions is None:
            reductions = [Reduction.MEAN, Reduction.MAX, Reduction.MIN]
        self.list_of_logs = []
        self.reductions = [Reduction.resolve(r) for r in reductions]
        self.prefix = ""

    def log(self, logs):
        self.list_of_logs.append(logs)

    def update(self, logs, index=-1):
        self.list_of_logs[index].update(logs)

    def collate(self, prefix=None, reductions=None):
        if reductions is not None:
            self.reductions = reductions

        if not self.list_of_logs:
            retval = {}
        else:
            retval = {
                f"{reduction.name.lower()}/{key}": value
                for reduction in self.reductions
                for key, value in self._collate_logs(self.list_of_logs, reduction).items()
            }

        self.list_of_logs = []
        return self._rename_logs(retval, prefix)

    def log_collate(self, logs, *args, **kwargs):
        self.log(logs)
        return self.collate(*args, **kwargs)

    def _rename_logs(self, logs, prefix=None):
        if prefix is not None:
            self.prefix = prefix

        if not logs:
            return {}

        return {f"{self.prefix}{key}": value for key, value in logs.items()}

    @staticmethod
    def _collate_logs(list_of_logs: list[dict], reduction=Reduction.MEAN):
        match reduction:
            case Reduction.MEAN | Reduction.SUM | Reduction.MAX | Reduction.MIN:
                keys = list(list_of_logs[0].keys())
                log_dict = {key: [log[key] for log in list_of_logs] for key in keys}
                retval = {
                    key: einops.reduce(log_dict[key], "b ... -> () ...", reduction.value)
                    for key in keys
                }
            case Reduction.LAST | Reduction.FIRST:
                retval = list_of_logs[reduction.value]
            case _:
                raise ValueError(f"Unknown reduction: {reduction}")

        return retval
