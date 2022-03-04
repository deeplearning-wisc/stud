""" Default Arguments """
from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C


# Load options that differ from the original detectron2 defaults. 

# ---------------------------------------------------------------------------- #
# Data Augmentation
# ---------------------------------------------------------------------------- #
# disable random flip during training
_C.DATALOADER.NO_FLIP = False
# offset range for pairs during pair sampling
_C.DATALOADER.PAIR_OFFSET_RANGE = 1


# ---------------------------------------------------------------------------- #
# Self supervised options
# ---------------------------------------------------------------------------- #
_C.MODEL.SS = CN()
_C.MODEL.SS.NAME = [
    "build_rotation_head"
]  # to be compatible with the exisiting configs; for more than one ss task, add bracket here
_C.MODEL.SS.FEAT_LEVEL = "res4"
_C.MODEL.SS.NUM_CLASSES = 4
_C.MODEL.SS.CROP_SIZE = 224
_C.MODEL.SS.LOSS_SCALE = 0.1
_C.MODEL.SS.RATIO = 1.0
_C.MODEL.SS.CLASS_FILE = "permutations/"
_C.MODEL.SS.ONLY = False
_C.MODEL.SS.JIGSAW = CN()
_C.MODEL.SS.JIGSAW.NORM = False
_C.MODEL.SS.COEF = -1.0
_C.MODEL.SS.ROI_THR = 0.8  #set 0.8 for training
_C.MODEL.SS.ROI_ALL = False  # use all ROI without score filtering and nms
_C.MODEL.SS.ENABLE_BATCH = False 
_C.MODEL.SS.BATCH_SIZE = 32
_C.MODEL.SS.ENERGY_WEIGHT = 1.0
_C.MODEL.FREEZE = 0
_C.MODEL.SS.SELECTED_FRAMES = 50
_C.MODEL.SS.CHEAP = 0
_C.MODEL.SS.LOSS = "normal"
_C.MODEL.SS.FILTERING1 = 0.1
_C.MODEL.SS.FILTERING2 = 0.1
_C.DATALOADER.SELCTED_NUMBER = 3
_C.DATALOADER.INTERVAL = 1



#
# _C.MODEL.VOVNET = CN()
#
# _C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
# _C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
#
# # Options: FrozenBN, GN, "SyncBN", "BN"
# _C.MODEL.VOVNET.NORM = "FrozenBN"
#
# _C.MODEL.VOVNET.OUT_CHANNELS = 256
#
# _C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256

# _C.MODEL.DLA = CN()
# _C.MODEL.DLA.CONV_BODY = "DLA60"
# _C.MODEL.DLA.NORM = 'FrozenBN'
# _C.MODEL.TRANSFORMER = CN()
# _C.MODEL.TRANSFORMER.DROP = 0.0
# _C.MODEL.TRANSFORMER.DROP_PATH = 0.1
# _C.MODEL.TRANSFORMER.NORM_EMBED = False
# _C.MODEL.TRANSFORMER.AVG_POOL = False
#
# _C.MODEL.TRANSFORMER.MSVIT = CN()
# _C.MODEL.TRANSFORMER.MSVIT.ARCH = 'l1,h3,d192,n1,s1,g1,f4,a0_l2,h6,d384,n10,s0,g1,f2,a0_l3,h6,d384,n1,s0,g1,f1,a0'
# _C.MODEL.TRANSFORMER.MSVIT.SHARE_W = True
# _C.MODEL.TRANSFORMER.MSVIT.ATTN_TYPE = 'longformerhand'
# _C.MODEL.TRANSFORMER.MSVIT.SHARE_KV = True
# _C.MODEL.TRANSFORMER.MSVIT.ONLY_GLOBAL = False
# _C.MODEL.TRANSFORMER.MSVIT.SW_EXACT = 0
# _C.MODEL.TRANSFORMER.MSVIT.LN_EPS = 1e-6
# _C.MODEL.TRANSFORMER.MSVIT.MODE = 0
# _C.MODEL.TRANSFORMER.MSVIT.REDRAW_INTERVAL = 1000
#
# _C.MODEL.TRANSFORMER.OUT_FEATURES = ()
#
# # input size should be patch_size x pos_embedding_size
# _C.INPUT.FIX_SIZE = ()
#
# # Optimizer.
# _C.SOLVER.OPTIMIZER = "ADAMW"
# _C.SOLVER.BACKBONE_MULTIPLIER = 1.0
#
# # Add LR multiplies to specific layers:
# # Use case:
# ##  SOLVER:
# ##     LR_MULTIPLIERS:
# ##          backbone: 0.1
# ##          embedding: 0.2
# ### it will apply 0.1 to layers with keyword 'backbone' and 0.2 to layers with keyword 'embedding'
# _C.SOLVER.LR_MULTIPLIERS = CN(new_allowed=True)


# # Apply deep stem
# _C.MODEL.RESNETS.DEEP_STEM = False
# # Apply avg after conv2 in the BottleBlock
# # When AVD=True, the STRIDE_IN_1X1 should be False
# _C.MODEL.RESNETS.AVD = False
# # Apply avg_down to the downsampling layer for residual path
# _C.MODEL.RESNETS.AVG_DOWN = False
#
# # Radix in ResNeSt
# _C.MODEL.RESNETS.RADIX = 1
# # Bottleneck_width in ResNeSt
# _C.MODEL.RESNETS.BOTTLENECK_WIDTH = 64



# _C.MODEL.TRANSFORMER = CN()
# _C.MODEL.TRANSFORMER.DROP = 0.0
# _C.MODEL.TRANSFORMER.DROP_PATH = 0.1
# _C.MODEL.TRANSFORMER.NORM_EMBED = False
# _C.MODEL.TRANSFORMER.AVG_POOL = False
#
# _C.MODEL.TRANSFORMER.MSVIT = CN()
# _C.MODEL.TRANSFORMER.MSVIT.ARCH = 'l1,h3,d192,n1,s1,g1,f4,a0_l2,h6,d384,n10,s0,g1,f2,a0_l3,h6,d384,n1,s0,g1,f1,a0'
# _C.MODEL.TRANSFORMER.MSVIT.SHARE_W = True
# _C.MODEL.TRANSFORMER.MSVIT.ATTN_TYPE = 'longformerhand'
# _C.MODEL.TRANSFORMER.MSVIT.SHARE_KV = True
# _C.MODEL.TRANSFORMER.MSVIT.ONLY_GLOBAL = False
# _C.MODEL.TRANSFORMER.MSVIT.SW_EXACT = 0
# _C.MODEL.TRANSFORMER.MSVIT.LN_EPS = 1e-6
# _C.MODEL.TRANSFORMER.MSVIT.MODE = 0
# _C.MODEL.TRANSFORMER.MSVIT.REDRAW_INTERVAL = 1000
#
# _C.MODEL.TRANSFORMER.OUT_FEATURES = ()
#
# # input size should be patch_size x pos_embedding_size
# _C.INPUT.FIX_SIZE = ()
#
# # Optimizer.
# _C.SOLVER.OPTIMIZER = "ADAMW"
# _C.SOLVER.BACKBONE_MULTIPLIER = 1.0
#
# # Add LR multiplies to specific layers:
# # Use case:
# ##  SOLVER:
# ##     LR_MULTIPLIERS:
# ##          backbone: 0.1
# ##          embedding: 0.2
# ### it will apply 0.1 to layers with keyword 'backbone' and 0.2 to layers with keyword 'embedding'
# _C.SOLVER.LR_MULTIPLIERS = CN(new_allowed=True)

# _C.MODEL.RESNETS.DEEP_STEM = False
# # Apply avg after conv2 in the BottleBlock
# # When AVD=True, the STRIDE_IN_1X1 should be False
# _C.MODEL.RESNETS.AVD = False
# # Apply avg_down to the downsampling layer for residual path
# _C.MODEL.RESNETS.AVG_DOWN = False
# # Radix in ResNeSt setting RADIX: 2
# _C.MODEL.RESNETS.RADIX = 2
# # Bottleneck_width in ResNeSt
# _C.MODEL.RESNETS.BOTTLENECK_WIDTH = 64

# _C.MODEL.RESNETS.DEPTH = 50
# _C.MODEL.RESNETS.OUT_FEATURES = ["res4"]  # res4 for C4 backbone, res2..5 for FPN backbone
#
# # Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
# _C.MODEL.RESNETS.NUM_GROUPS = 1
#
# # Options: FrozenBN, GN, "SyncBN", "BN"
# _C.MODEL.RESNETS.NORM = "FrozenBN"
#
# # Baseline width of each group.
# # Scaling this parameters will scale the width of all bottleneck layers.
# _C.MODEL.RESNETS.WIDTH_PER_GROUP = 64
#
# # Place the stride 2 conv on the 1x1 filter
# # Use True only for the original MSRA ResNet; use False for C2 and Torch models
# _C.MODEL.RESNETS.STRIDE_IN_1X1 = True
#
# # Apply dilation in stage "res5"
# _C.MODEL.RESNETS.RES5_DILATION = 1
#
# # Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet
# # For R18 and R34, this needs to be set to 64
# _C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
# _C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
#
# # Apply Deformable Convolution in stages
# # Specify if apply deform_conv on Res2, Res3, Res4, Res5
# _C.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
# # Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
# # Use False for DeformableV1.
# _C.MODEL.RESNETS.DEFORM_MODULATED = False
# # Number of groups in deformable conv.
# _C.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1
#
#
# # Apply deep stem
# _C.MODEL.RESNETS.DEEP_STEM = False
# # Apply avg after conv2 in the BottleBlock
# # When AVD=True, the STRIDE_IN_1X1 should be False
# _C.MODEL.RESNETS.AVD = False
# # Apply avg_down to the downsampling layer for residual path
# _C.MODEL.RESNETS.AVG_DOWN = False
#
# # Radix in ResNeSt
# _C.MODEL.RESNETS.RADIX = 1
# # Bottleneck_width in ResNeSt
# _C.MODEL.RESNETS.BOTTLENECK_WIDTH = 64
# ---------------------------------------------------------------------------- #
# Specific testing time training options
# ---------------------------------------------------------------------------- #
# _C.TTT = CN()
# _C.TTT.ENABLE = False
# _C.TTT.STEPS = 10
# _C.TTT.SAVE_ROI = False
# _C.TTT.MAX_ITERS = 10
# _C.TTT.BATCH_SIZE = 32
# _C.TTT.RANDOM_BATCH = False
# _C.TTT.INTERVAL = 1
# _C.TTT.ENABLE_BATCH = False
# _C.TTT.NO_BP = False
# _C.TTT.SAVE_BN = False
# _C.TTT.USE_BN = False

# # use full ROI proposal during testing
# _C.TTT.ROI_THR = 0.8  # use 0.8 for training
# _C.TTT.ROI_ALL = False  # use all ROI without score filtering and nms


# _C.TTT.REVERSE = False
# _C.TTT.SS_THRESHOLD = 0.0
# _C.TTT.EXTRA_STEPS = 0
# _C.TTT.WARMUP_ITERS = 0
# _C.TTT.ORACLE = False
# _C.TTT.LAST_OCC = False
# _C.TTT.ADAPT = False
# _C.TTT.ORACLE = False
# _C.TTT.RESET = False
# _C.TTT.CLASS_WEIGHT = False
# _C.TTT.ALL_WEIGHT = False

# _C.CONST = CN()
# _C.CONST.TOPK = 1
# _C.CONST.STEP = 1

