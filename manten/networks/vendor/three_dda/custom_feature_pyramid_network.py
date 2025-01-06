from collections import OrderedDict
from collections.abc import Callable

import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops.misc import Conv2dNormActivation


class ExtraFPNBlock(nn.Module):
    """
    Base class for the extra block in the FPN.

    Args:
        results (List[Tensor]): the result of the FPN
        x (List[Tensor]): the original feature maps
        names (List[str]): the names for each one of the
            original feature maps

    Returns:
        results (List[Tensor]): the extended set of results
            of the FPN
        names (List[str]): the extended set of names for the results
    """

    def forward(
        self,
        results: list[Tensor],
        x: list[Tensor],
        names: list[str],
    ) -> tuple[list[Tensor], list[str]]:
        pass


class UsageAwareMemoryEfficientFeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps.
    See torchvision.ops.FeaturePyramidNetwork for more details.

    Compared to the original implementation, this module accepts the
    `sampled_pyramid` argument, which is a list of integers that
    specifies which feature maps are used in the output. This is used
    to reduce the memory footprint and computation time of the FPN. It
    also helps in DDP training, as the unused parameters are not created
    in the first place.
    """

    def __init__(
        self,
        in_channels_list: list[int],
        out_channels: int,
        sampled_pyramid: list[int],
        extra_blocks: ExtraFPNBlock | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ):
        super().__init__()
        self.sampled_pyramid = sampled_pyramid
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        max_sampled_pyramid = max(sampled_pyramid)
        for idx, in_channels in enumerate(in_channels_list):
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")

            # these cascade from top to bottom, for output level n only innerblocks m with m >= n are used
            inner_block_module = (
                None
                if (idx < max_sampled_pyramid)
                else Conv2dNormActivation(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    norm_layer=norm_layer,
                    activation_layer=None,
                )
            )
            self.inner_blocks.append(inner_block_module)

            # these are only used in their own outputs, so sampled_pyramid directly
            layer_block_module = (
                None
                if (idx not in sampled_pyramid)
                else Conv2dNormActivation(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    norm_layer=norm_layer,
                    activation_layer=None,
                )
            )
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if extra_blocks is not None and not isinstance(extra_blocks, ExtraFPNBlock):
            raise TypeError(
                f"extra_blocks should be of type ExtraFPNBlock not {type(extra_blocks)}"
            )
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                if module is not None:
                    out = module(x)
                else:
                    out = None
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                if module is not None:
                    out = module(x)
                else:
                    out = None
        return out

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from the highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        results = []
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)  # 4
        results.append(self.get_result_from_layer_blocks(last_inner, -1))
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)  # 3, 2, 1, 0
            if inner_lateral is None:
                # break according to how we use the module, see #sample_pyramid
                results.insert(0, None)
                continue
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")

            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict(list(zip(names, results, strict=True)))

        return out
