# from ..common.optim import SGD as optimizer
# from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
# from ..common.data.coco import dataloader
# from ..common.models.mask_rcnn_fpn import model
# from ..common.train import train

# from detectron2.config import LazyCall as L
from .regnet_model import RegNet
from .regnet_model import SimpleStem, ResBottleneckBlock

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

from detectron2.layers import (
    Conv2d,
    DeformConv,
    FrozenBatchNorm2d,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

# Replace default ResNet with RegNetY-4GF from the DDS paper. Config source:
# https://github.com/facebookresearch/pycls/blob/2c152a6e5d913e898cca4f0a758f41e6b976714d/configs/dds_baselines/regnety/RegNetY-4.0GF_dds_8gpu.yaml#L4-L10  # noqa
# model.backbone.bottom_up = L(RegNet)(
#     stem_class=SimpleStem,
#     stem_width=32,
#     block_class=ResBottleneckBlock,
#     depth=22,
#     w_a=31.41,
#     w_0=96,
#     w_m=2.24,
#     group_width=64,
#     se_ratio=0.25,
#     freeze_at=2,
#     norm="FrozenBN",
#     out_features=["s1", "s2", "s3", "s4"],
# )
# model.pixel_std = [57.375, 57.120, 58.395]
#
# optimizer.weight_decay = 5e-5
# train.init_checkpoint = (
#     "https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906838/RegNetY-4.0GF_dds_8gpu.pyth"
# )
# # RegNets benefit from enabling cudnn benchmark mode
# train.cudnn_benchmark = True

@BACKBONE_REGISTRY.register()
def build_regnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = RegNet(
            stem_class=SimpleStem,
            stem_width=32,
            block_class=ResBottleneckBlock,
            depth=22,
            w_a=31.41,
            w_0=96,
            w_m=2.24,
            group_width=64,
            se_ratio=0.25,
            freeze_at=2,
            norm="FrozenBN",
            out_features=["s1", "s2", "s3", "s4"],
        )
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_regnetx_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = RegNet(
        stem_class=SimpleStem,
        stem_width=32,
        block_class=ResBottleneckBlock,
        depth=23,
        w_a=38.65,
        w_0=96,
        w_m=2.43,
        group_width=40,
        freeze_at=2,
        norm="FrozenBN",
        out_features=["s1", "s2", "s3", "s4"],
)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone