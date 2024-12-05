class LogCollator:
    def __init__(self, reductions=["mean", "max", "min"]):
        self.list_of_logs = []
        self.prefix = ""
        self.reductions = reductions

    def log(self, logs):
        self.list_of_logs.append(logs)

    def update(self, logs):
        self.list_of_logs[-1].update(logs)

    def collate(self, prefix=None, reductions=None):
        if prefix is not None:
            self.prefix = prefix
        if reductions is not None:
            self.reductions = reductions

        if not self.list_of_logs:
            retval = {}
        elif len(self.list_of_logs) == 1:
            retval = {
                f"{self.prefix}{key}": value for key, value in self.list_of_logs[0].items()
            }
        else:
            retval = {
                f"{self.prefix}{key}": value
                for reduction in self.reductions
                for key, value in self._collate_logs(self.list_of_logs, reduction).items()
            }

        self.list_of_logs = []
        return retval

    def log_collate(self, logs, *args, **kwargs):
        self.log(logs)
        return self.collate(*args, **kwargs)

    @staticmethod
    def _collate_logs(list_of_logs, reduction="mean", rename=True):
        if reduction == "mean":
            retval = {
                key: sum(log[key] for log in list_of_logs) / len(list_of_logs)
                for key in list_of_logs[0]
            }
        elif reduction == "sum":
            retval = {key: sum(log[key] for log in list_of_logs) for key in list_of_logs[0]}
        elif reduction == "max":
            retval = {key: max(log[key] for log in list_of_logs) for key in list_of_logs[0]}
        elif reduction == "min":
            retval = {key: min(log[key] for log in list_of_logs) for key in list_of_logs[0]}
        elif reduction == "last":
            retval = list_of_logs[-1]
        elif reduction == "first":
            retval = list_of_logs[0]
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        if rename:
            retval = {f"{reduction}/{key}": value for key, value in retval.items()}
        return retval
