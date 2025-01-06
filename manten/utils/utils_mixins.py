def state_dict_mixin_factory(*attrs):
    class StateDictMixin:
        def state_dict(self):
            parent_state = super().state_dict() if hasattr(super(), "state_dict") else {}
            current_state = {attr: getattr(self, attr) for attr in attrs}
            return {**parent_state, **current_state}

        def load_state_dict(self, state_dict):
            parent_state = {k: v for k, v in state_dict.items() if k not in attrs}
            if hasattr(super(), "load_state_dict"):
                super().load_state_dict(parent_state)
            for attr in attrs:
                setattr(self, attr, state_dict[attr])

    return StateDictMixin


def shallow_copy_mixin_factory(*attrs):
    class ShallowCopyMixin:
        def copy(self):
            cpy = super().copy() if hasattr(super(), "copy") else self.__class__()
            for attr in attrs:
                setattr(cpy, attr, getattr(self, attr))
            return cpy

    return ShallowCopyMixin
