def with_name_resolution(enum_cls):
    def resolve(cls, name):
        if isinstance(name, cls):
            return name
        for member in cls:
            if member.name.lower() == name.lower():
                return member
        raise ValueError(f"No matching member found for name: {name}")

    enum_cls.resolve = classmethod(resolve)
    return enum_cls
