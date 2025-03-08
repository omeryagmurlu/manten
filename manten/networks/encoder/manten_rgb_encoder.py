from collections.abc import Callable
from typing import Literal

import einops
import optree
import torchvision
from torch import nn

from manten.utils.utils_pytree import tree_rearrange


def transpose_array_of_dicts(array_of_dicts):
    return {key: [d[key] for d in array_of_dicts] for key in array_of_dicts[0]}


class MantenRGBTransforms(nn.Module):
    def __init__(  # noqa: PLR0912
        self,
        *,
        resize: tuple[int, int] | dict[str, tuple[int, int]] | None = None,
        crop: tuple[int, int] | dict[str, tuple[int, int]] | None = None,
        crop_type: Literal["center", "random"] = "center",
        imagenet_norm: bool = False,
    ):
        super().__init__()

        if isinstance(resize, tuple):
            self.resize = torchvision.transforms.Resize(size=resize)
        elif isinstance(resize, dict):
            self.resize = {
                key: torchvision.transforms.Resize(size=resize)
                for key, resize in resize.items()
            }
        else:
            self.resize = None

        if isinstance(crop, tuple):
            self.crop_test = torchvision.transforms.CenterCrop(size=crop)
            if crop_type == "random":
                self.crop_train = torchvision.transforms.RandomCrop(size=crop)
            elif crop_type == "center":
                self.crop_train = torchvision.transforms.CenterCrop(size=crop)
            else:
                raise ValueError("Invalid crop type")
        elif isinstance(crop, dict):
            self.crop_test = {
                key: torchvision.transforms.CenterCrop(size=crop)
                for key, crop in crop.items()
            }
            if crop_type == "random":
                self.crop_train = {
                    key: torchvision.transforms.RandomCrop(size=crop)
                    for key, crop in crop.items()
                }
            elif crop_type == "center":
                self.crop_train = {
                    key: torchvision.transforms.CenterCrop(size=crop)
                    for key, crop in crop.items()
                }
            else:
                raise ValueError("Invalid crop type")
        else:
            self.crop_test = None
            self.crop_train = None

        if imagenet_norm:
            self.normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize = None

    def forward(self, rgb_dict):
        if isinstance(self.resize, dict):
            rgb_dict = {key: self.resize[key](rgb_dict[key]) for key in rgb_dict}
        elif self.resize is not None:
            rgb_dict = {key: self.resize(rgb_dict[key]) for key in rgb_dict}

        if self.training:
            if isinstance(self.crop_train, dict):
                rgb_dict = {key: self.crop_train[key](rgb_dict[key]) for key in rgb_dict}
            elif self.crop_train is not None:
                rgb_dict = self.crop_train(rgb_dict)
        else:  # noqa: PLR5501
            if isinstance(self.crop_test, dict):
                rgb_dict = {key: self.crop_test[key](rgb_dict[key]) for key in rgb_dict}
            elif self.crop_test is not None:
                rgb_dict = self.crop_test(rgb_dict)

        if self.normalize is not None:
            rgb_dict = {key: self.normalize(rgb_dict[key]) for key in rgb_dict}

        return rgb_dict


class MantenRGBEncoder(nn.Module):
    def __init__(
        self,
        *,
        rgb_shape: dict[str, tuple[int]],
        rgb_model: Callable[..., nn.Module],
        share_model: bool = False,
        transform_kwargs=None,
        merge_model_invocations_if_shared: bool = False,
        combine_features: Literal[
            "global_only", "local_only", "global_and_local"
        ] = "global_and_local",
        combine_cameras: Literal[
            "concat_channels", "stack_after_batch", "op_channels", "cross_attention"
        ] = "stack_after_batch",
        op_last_dim_operation: Literal["sum", "mean", "max"] = "max",
        select_local_feature: str | None = None,
    ):
        if transform_kwargs is None:
            transform_kwargs = {}
        super().__init__()

        self.share_model = share_model
        if self.share_model:
            self.rgb_model = rgb_model()
            self.merge_model_invocations = merge_model_invocations_if_shared
        else:
            self.rgb_models = nn.ModuleDict({key: rgb_model() for key in rgb_shape})

        self.transforms = MantenRGBTransforms(**transform_kwargs)

        self.select_local_feature = select_local_feature
        self.combine_features = combine_features
        self.combine_cameras = combine_cameras
        if self.combine_cameras == "cross_attention":
            # cross attention with: seq_len learnable params as query, no positional encoding, key value from input
            raise NotImplementedError("Cross attention not implemented yet")
        if self.combine_cameras == "op_channels":
            raise NotImplementedError("Operation on last dim not implemented yet")

    def forward(self, rgb_obs):
        features = self.forward_model(rgb_obs)
        return self.forward_combine(features)

    def forward_model(self, rgb_obs):
        shape = next(iter(rgb_obs.values())).shape
        B = shape[0]

        assert len(shape) == 4, "RGB shape must be (B, C, H, W)"  # noqa: PLR2004

        if self.transforms is not None:
            rgb_obs = self.transforms(rgb_obs)

        if self.share_model:
            if (not self.merge_model_invocations) or len(rgb_obs) == 1:
                rgb_features = {key: self.rgb_model(rgb_obs) for key in rgb_obs}
            else:
                len_cams = len(rgb_obs)
                rgb_features = einops.rearrange(
                    list(rgb_obs.values()), "cam b ... -> (cam b) ..."
                )
                rgb_features = self.rgb_model(rgb_features)
                # -> [(cam b) ...] / {k: [(cam b) ...]}
                rgb_features = tree_rearrange(
                    rgb_features,
                    "(cam b) ... -> cam b ...",
                    b=B,
                    cam=len_cams,
                )  # -> [cam b t ...] / {k: [cam b t ...]}
                rgb_features = {
                    key: optree.tree_map(lambda x, *, idx=idx: x[idx], rgb_features)
                    for idx, key in enumerate(rgb_obs.keys())
                }  # -> {cam: [b t ...] / {k: [b t ...]} }
        else:
            rgb_features = {key: self.rgb_models[key](rgb_obs[key]) for key in rgb_obs}

        return rgb_features  # dict[str-cam, tuple[tensor, dict[str-featname, tensor]]]

    def forward_combine(self, rgb_features):
        if "global" in self.combine_features:
            global_features = einops.rearrange(
                [v[0] for v in rgb_features.values()], "cam b ... -> b cam ..."
            )
            if self.combine_cameras == "concat_channels":
                global_features = einops.rearrange(global_features, "b cam d -> b 1 (cam d)")
            elif self.combine_cameras == "stack_after_batch":
                pass
            else:
                raise NotImplementedError

        if "local" in self.combine_features:
            local_features = {
                k: einops.rearrange(v, "cam b c ... -> b cam (...) c")
                for k, v in transpose_array_of_dicts(
                    [v[1] for v in rgb_features.values()]
                ).items()
            }
            if self.combine_cameras == "concat_channels":
                local_features = {
                    k: einops.rearrange(v, "b cam n d -> b n (cam d)")
                    for k, v in local_features.items()
                }
            elif self.combine_cameras == "stack_after_batch":
                local_features = {
                    k: einops.rearrange(v, "b cam n d -> b (cam n) d")
                    for k, v in local_features.items()
                }
            else:
                raise NotImplementedError

            if self.select_local_feature is not None:
                local_features = local_features[self.select_local_feature]

        if self.combine_features == "global_only":
            return global_features, None
        elif self.combine_features == "local_only":
            return None, local_features
        elif self.combine_features == "global_and_local":
            return global_features, local_features

        raise ValueError("Invalid combine_features value")
