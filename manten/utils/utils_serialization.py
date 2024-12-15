import pickle
from dataclasses import dataclass

import numpy as np
from optree import tree_map

NP_IS_OLD_CMPD = np.__version__.startswith("1")


@dataclass
class SerializedObject:
    """Used to fool optree.tree_map leaf detection so that we don't need to provide
    a custom is_leaf predicate and instead use the default and efficient C++ impl."""

    serialize_type: str
    data: bytes


class MantenAgentSerialization:
    @staticmethod
    def serialize_mapper(obj):
        # if isinstance(obj, torch.Tensor):
        #     # no problems so far with this
        #     return SerializedObject(serialize_type="torch.Tensor", data=obj)
        if not NP_IS_OLD_CMPD and isinstance(obj, np.ndarray):
            # need to serialize when new -> old numpy
            return SerializedObject(serialize_type="np.ndarray", data=obj.tolist())
        return obj

    @staticmethod
    def deserialize_mapper(obj):
        if isinstance(obj, SerializedObject):  # noqa: SIM102
            # if obj.serialize_type == "torch.Tensor":
            #     return obj.data
            if obj.serialize_type == "np.ndarray":
                return np.array(obj.data)
        return obj

    @staticmethod
    def deserialize(content):
        return tree_map(
            MantenAgentSerialization.deserialize_mapper,
            pickle.loads(content),  # noqa: S301
        )

    @staticmethod
    def serialize(obj):
        return pickle.dumps(tree_map(MantenAgentSerialization.serialize_mapper, obj))
