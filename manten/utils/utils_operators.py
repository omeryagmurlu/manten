def is_sortable(cls):
    return cls.__lt__ != object.__lt__ or cls.__gt__ != object.__gt__


def does_support_arithmetic(cls):
    return (
        cls.__add__ != object.__add__
        or cls.__sub__ != object.__sub__
        or cls.__mul__ != object.__mul__
        or cls.__truediv__ != object.__truediv__
        or cls.__floordiv__ != object.__floordiv__
        or cls.__mod__ != object.__mod__
        or cls.__pow__ != object.__pow__
    )
