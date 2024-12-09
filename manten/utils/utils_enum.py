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
