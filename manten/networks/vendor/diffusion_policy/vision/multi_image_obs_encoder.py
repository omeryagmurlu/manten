import copy

import torch
import torchvision
from torch import nn

from manten.networks.utils.mixins import ModuleAttrMixin
from manten.networks.vendor.diffusion_policy.vision.crop_randomizer import CropRandomizer
from manten.utils.utils_pytorch import replace_submodules


class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        rgb_shape: dict,
        rgb_model: nn.Module | dict[str, nn.Module],
        resize_shape: tuple[int, int] | dict[str, tuple] | None = None,
        crop_shape: tuple[int, int] | dict[str, tuple] | None = None,
        random_crop: bool = True,
        # replace BatchNorm with GroupNorm
        use_group_norm: bool = False,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = False,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm: bool = False,
    ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        # rgb shape meta: key -> shape/shape in list form
        rgb_shape = {k: torch.Size(v) for k, v in rgb_shape.items()}

        rgb_keys = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = {}

        # handle sharing vision backbone
        if share_rgb_model:
            if not isinstance(rgb_model, nn.Module):
                raise ValueError("Must provide a single model when sharing")
            this_model = rgb_model
            if use_group_norm:
                this_model = replace_submodules(
                    root_module=this_model,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=x.num_features // 16, num_channels=x.num_features
                    ),
                )

        for key, shape in rgb_shape.items():
            # shape = tuple(attr["shape"])
            # typ = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            rgb_keys.append(key)
            # configure model for this key
            if not share_rgb_model:
                if isinstance(rgb_model, dict):
                    # have provided model for each key
                    this_model = rgb_model[key]
                else:
                    if not isinstance(rgb_model, nn.Module):
                        raise TypeError("Must provide a single model or dict of models")
                    # have a copy of the rgb model
                    this_model = copy.deepcopy(rgb_model)

                if use_group_norm:
                    this_model = replace_submodules(
                        root_module=this_model,
                        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                        func=lambda x: nn.GroupNorm(
                            num_groups=x.num_features // 16, num_channels=x.num_features
                        ),
                    )

            key_model_map[key] = this_model

            # configure resize
            input_shape = shape
            this_resizer = nn.Identity()
            if resize_shape is not None:
                if isinstance(resize_shape, dict):
                    h, w = resize_shape[key]
                else:
                    h, w = resize_shape
                this_resizer = torchvision.transforms.Resize(size=(h, w))
                input_shape = (shape[0], h, w)

            # configure randomizer
            this_randomizer = nn.Identity()
            if crop_shape is not None:
                if isinstance(crop_shape, dict):
                    h, w = crop_shape[key]
                else:
                    h, w = crop_shape
                if random_crop:
                    this_randomizer = CropRandomizer(
                        input_shape=input_shape,
                        crop_height=h,
                        crop_width=w,
                        num_crops=1,
                        pos_enc=False,
                    )
                else:
                    this_normalizer = torchvision.transforms.CenterCrop(size=(h, w))
            # configure normalizer
            this_normalizer = nn.Identity()
            if imagenet_norm:
                this_normalizer = torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )

            this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
            key_transform_map[key] = this_transform
        rgb_keys = sorted(rgb_keys)

        self.rgb_shape = rgb_shape
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.key_shape_map = key_shape_map

    def forward(self, obs_dict):
        batch_size = None
        features = []
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map["rgb"](imgs)
            # (N,B,D)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature, 0, 1)
            # (B,N*D)
            assert batch_size is not None
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        # # process lowdim input
        # for key in self.low_dim_keys:
        #     data = obs_dict[key]
        #     if batch_size is None:
        #         batch_size = data.shape[0]
        #     else:
        #         assert batch_size == data.shape[0]
        #     assert data.shape[1:] == self.key_shape_map[key]
        #     features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result

    # @torch.no_grad()
    # def output_shape(self):
    #     example_obs_dict = {}
    #     batch_size = 1
    #     for key, shape in self.rgb_shape.items():
    #         this_obs = torch.zeros((batch_size, *shape), dtype=self.dtype, device=self.device)
    #         example_obs_dict[key] = this_obs
    #     example_output = self.forward(example_obs_dict)
    #     output_shape = example_output.shape[1:]
    #     return output_shape
