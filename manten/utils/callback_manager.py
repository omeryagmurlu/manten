# class Callback:
#     def at_train_batch_end(self, *, epoch, step, batch, log_aggregator, batch_result):
#         pass

#     def at_validation_batch_end(self, *, epoch, step, batch, log_aggregator, batch_result):
#         pass

#     def at_eval_batch_end(self, *, epoch, step, batch, log_aggregator, batch_result):
#         pass

#     def at_batch_end(self, *, epoch, step, batch, log_aggregator, batch_result):
#         pass

#     def at_train_batch_start(self, *, epoch, step, batch, log_aggregator):
#         pass

#     def at_validation_batch_start(self, *, epoch, step, batch, log_aggregator):
#         pass

#     def at_eval_batch_start(self, *, epoch, step, batch, log_aggregator):
#         pass

#     def at_batch_start(self, *, epoch, step, batch, log_aggregator):
#         pass

#     def at_train_epoch_end(self, *, epoch, log_aggregator):
#         pass

#     def at_validation_epoch_end(self, *, epoch, log_aggregator):
#         pass

#     def at_eval_epoch_end(self, *, epoch, log_aggregator):
#         pass

#     def at_epoch_end(self, *, epoch, log_aggregator):
#         pass

#     def at_train_epoch_start(self, *, epoch, log_aggregator):
#         pass

#     def at_validation_epoch_start(self, *, epoch, log_aggregator):
#         pass

#     def at_eval_epoch_start(self, *, epoch, log_aggregator):
#         pass

#     def at_epoch_start(self, *, epoch, log_aggregator):
#         pass


# test_prefix = "at_train"
# CB_LOC = [name for name in dir(Callback) if name.startswith(test_prefix)]
# CB_SPLITS = ["train", "validation", "eval"]

# PARENTS = {f"at_{split}_{loc}": f"at_{loc}" for loc in CB_LOC for split in CB_SPLITS}
# INVOCATIONS = [f"at_{split}_{loc}" for loc in CB_LOC for split in CB_SPLITS] + [
#     f"at_{loc}" for loc in CB_LOC
# ]


# class CallbackManager:
#     def __init__(self, callbacks=None):
#         if callbacks is None:
#             callbacks = []
#         self.callbacks = callbacks

#     def register(self, callback):
#         self.callbacks.append(callback)

#     def invoke(self, cb_name, **kwargs):
#         if cb_name not in INVOCATIONS:
#             return
#         for callback in self.callbacks:
#             getattr(callback, PARENTS[cb_name])(**kwargs)
#             getattr(callback, cb_name)(**kwargs)

#     @staticmethod
#     def map(split):
#         return {f"at_{loc}": f"at_{split}_{loc}" for loc in CB_LOC}
