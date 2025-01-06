import functools
from typing import TypeVar

import numpy as np
import torch

import manten.networks.utils.pytorch3d_transforms as pt

T = TypeVar("T", np.ndarray, torch.Tensor)


class RotationTransformer:
    valid_reps = ("axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix")
    dims = (3, 3, 4, 6, None)

    def __init__(
        self,
        from_rep,
        to_rep,
        from_convention=None,
        to_convention=None,
    ):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        forward_funcs = []
        inverse_funcs = []

        if from_rep != "matrix":
            funcs = [
                getattr(pt, f"{from_rep}_to_matrix"),
                getattr(pt, f"matrix_to_{from_rep}"),
            ]
            if from_convention is not None:
                funcs = [
                    functools.partial(func, convention=from_convention) for func in funcs
                ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(pt, f"{to_rep}_to_matrix")]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs.reverse()

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs
        self.from_rep = from_rep
        self.to_rep = to_rep

    @staticmethod
    def _apply_funcs(x: T, funcs: list) -> T:
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        else:
            x_ = x
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        if isinstance(x, np.ndarray):
            return x_.numpy()
        return x_

    def forward(self, x: T) -> T:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: T) -> T:
        return self._apply_funcs(x, self.inverse_funcs)

    @property
    def to_dim(self):
        td = self.dims[self.valid_reps.index(self.to_rep)]
        if td is None:
            raise ValueError("Cannot determine dimension of the target representation")
        return td

    @property
    def from_dim(self):
        fd = self.dims[self.valid_reps.index(self.from_rep)]
        if fd is None:
            raise ValueError("Cannot determine dimension of the source representation")
        return fd


def test():
    tf = RotationTransformer("axis_angle", "rotation_6d")

    rotvec = np.random.uniform(-2 * np.pi, 2 * np.pi, size=(1000, 3))  # noqa: NPY002
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation

    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    dist = diff.magnitude()
    assert isinstance(dist, np.ndarray)
    assert dist.max() < 1e-7  # noqa: PLR2004

    tf = RotationTransformer("rotation_6d", "matrix")
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)  # noqa: NPY002
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
    # rotaiton_6d will be normalized to rotation matrix
