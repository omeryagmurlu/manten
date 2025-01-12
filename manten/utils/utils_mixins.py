def recursive_state_dict_mixin_factory(*attrs):
    class RecursiveStateDictMixin:
        def state_dict(self):
            parent_state = super().state_dict() if hasattr(super(), "state_dict") else {}
            current_state = {}
            for attr in attrs:
                orig_attr_val = getattr(self, attr)
                if hasattr(orig_attr_val, "state_dict"):
                    attr_state = orig_attr_val.state_dict()
                else:
                    attr_state = orig_attr_val
                current_state[attr] = attr_state
            return {**parent_state, **current_state}

        def load_state_dict(self, state_dict):
            parent_state = {k: v for k, v in state_dict.items() if k not in attrs}
            current_state = {k: v for k, v in state_dict.items() if k in attrs}
            if hasattr(super(), "load_state_dict"):
                super().load_state_dict(parent_state)
            for attr, attr_state in current_state.items():
                orig_attr_val = getattr(self, attr)
                if hasattr(orig_attr_val, "load_state_dict"):
                    orig_attr_val.load_state_dict(
                        attr_state
                    )  # it is expected that you copy first, then load
                else:
                    setattr(self, attr, attr_state)

    return RecursiveStateDictMixin


def recursive_copy_mixin_factory(*attrs):
    class RecursiveCopyMixin:
        def copy(self):
            cpy = super().copy() if hasattr(super(), "copy") else self.__class__()
            for attr in attrs:
                orig_attr_val = getattr(self, attr)
                if hasattr(orig_attr_val, "copy"):
                    cpy_attr_val = orig_attr_val.copy()
                else:
                    cpy_attr_val = orig_attr_val
                setattr(cpy, attr, cpy_attr_val)
            return cpy

    return RecursiveCopyMixin
