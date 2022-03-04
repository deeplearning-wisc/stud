from detectron2.config import CfgNode as CN


def add_vit_config(cfg):
    """
    Add config for VIT.
    """
    cfg.MODEL.TRANSFORMER = CN()
    cfg.MODEL.TRANSFORMER.DROP = 0.0
    cfg.MODEL.TRANSFORMER.DROP_PATH = 0.1
    cfg.MODEL.TRANSFORMER.NORM_EMBED = False
    cfg.MODEL.TRANSFORMER.AVG_POOL = False

    cfg.MODEL.TRANSFORMER.MSVIT = CN()
    cfg.MODEL.TRANSFORMER.MSVIT.ARCH = 'l1,h3,d192,n1,s1,g1,f4,a0_l2,h6,d384,n10,s0,g1,f2,a0_l3,h6,d384,n1,s0,g1,f1,a0'
    cfg.MODEL.TRANSFORMER.MSVIT.SHARE_W = True
    cfg.MODEL.TRANSFORMER.MSVIT.ATTN_TYPE = 'longformerhand'
    cfg.MODEL.TRANSFORMER.MSVIT.SHARE_KV = True
    cfg.MODEL.TRANSFORMER.MSVIT.ONLY_GLOBAL = False
    cfg.MODEL.TRANSFORMER.MSVIT.SW_EXACT = 0
    cfg.MODEL.TRANSFORMER.MSVIT.LN_EPS = 1e-6
    cfg.MODEL.TRANSFORMER.MSVIT.MODE = 0
    cfg.MODEL.TRANSFORMER.MSVIT.REDRAW_INTERVAL = 1000

    cfg.MODEL.TRANSFORMER.OUT_FEATURES = ()

    # input size should be patch_size x pos_embedding_size
    cfg.INPUT.FIX_SIZE = ()

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # Add LR multiplies to specific layers:
    # Use case:
    ##  SOLVER:
    ##     LR_MULTIPLIERS:
    ##          backbone: 0.1
    ##          embedding: 0.2
    ### it will apply 0.1 to layers with keyword 'backbone' and 0.2 to layers with keyword 'embedding'
    cfg.SOLVER.LR_MULTIPLIERS = CN(new_allowed=True)
