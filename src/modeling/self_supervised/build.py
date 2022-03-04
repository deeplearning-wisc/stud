from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

SSHEAD_REGISTRY = Registry("SSHEAD")
SSHEAD_REGISTRY.__doc__ = """
return self-supervised head 
"""


def build_ss_head(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    ss_name = cfg.MODEL.SS.NAME

    ss_head = [SSHEAD_REGISTRY.get(name)(cfg, input_shape) for name in ss_name]
    assert len(ss_head) != 0
    return ss_head
