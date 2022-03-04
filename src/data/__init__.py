from .builtin import (
    # register_all_waymo,

    register_all_bdd_tracking,
    register_all_coco,
    register_coco_ood_wrt_bdd,
    register_vis_dataset,


)

from .pair_sampler import PairTrainingSampler, PairDataLoader
from .pair_fix_sampler import PairFixTrainingSampler, PairFixDataLoader
from .pair_all_sampler import PairAllTrainingSampler, PairAllDataLoader
from .pair_sampler_multi_random import PairTrainingMultiRandomSampler, PairMultirandomDataLoader
from .pair_sampler_multi_interval import PairTrainingMultiIntervalSampler, PairDataIntervalLoader

# from .common import MapDataset

from .build import build_detection_train_loader, get_detection_dataset_dicts

# Register them all under "./datasets"
# register_all_bdd100k()
# register_all_waymo()

#
register_all_bdd_tracking()
register_all_coco()
register_coco_ood_wrt_bdd()
register_vis_dataset()
