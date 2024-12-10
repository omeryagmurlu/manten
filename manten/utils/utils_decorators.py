from enum import Enum
from typing import TypeVar

E = TypeVar("E", bound=Enum)


def with_name_resolution(enum_cls: E):
    def resolve(that: E, name) -> E:
        if isinstance(name, that):
            return name
        for member in that:
            if member.name.lower() == name.lower():
                return member
        raise ValueError(f"No matching member found for name: {name}")

    enum_cls.resolve = classmethod(resolve)
    return enum_cls


def with_state_dict(*attrs):
    def decorator(cls):
        def state_dict(self):
            parent_state = (
                super(cls, self).state_dict()
                if hasattr(super(cls, self), "state_dict")
                else {}
            )
            current_state = {attr: getattr(self, attr) for attr in attrs}
            return {**parent_state, **current_state}

        def load_state_dict(self, state_dict):
            parent_state = {k: v for k, v in state_dict.items() if k not in attrs}
            if hasattr(super(cls, self), "load_state_dict"):
                super(cls, self).load_state_dict(parent_state)
            for attr in attrs:
                setattr(self, attr, state_dict[attr])

        cls.state_dict = state_dict
        cls.load_state_dict = load_state_dict
        return cls

    return decorator


def with_shallow_copy(*attrs):
    def decorator(cls):
        def copy(self):
            cpy = (
                super(cls, self).copy()
                if hasattr(super(cls, self), "copy")
                else self.__class__()
            )
            for attr in attrs:
                setattr(cpy, attr, getattr(self, attr))
            return cpy

        cls.copy = copy
        return cls

    return decorator
