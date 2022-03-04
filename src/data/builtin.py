import json
import os
import os.path as osp
from collections import defaultdict

from detectron2.data import DatasetCatalog
from .coco import register_coco_instances  # use customized data register

GPU=0

if GPU==0:
    __all__ = [
        "register_all_bdd_tracking",
        "register_all_waymo",
    ]


    def load_json(filename):
        with open(filename, "r") as fp:
            reg_file = json.load(fp)
        return reg_file


    # ==== Predefined datasets and splits for BDD100K ==========
    # BDD100K MOT set domain splits.
    _PREDEFINED_SPLITS_BDDT = {
        "bdd_tracking_2k": {
            "bdd_tracking_2k_train": (
                "bdd100k/images/track/train",
                "bdd100k/labels/track/bdd100k_mot_train_coco.json",
            ),
            "bdd_tracking_2k_val": (
                "bdd100k/images/track/val",
                "bdd100k/labels/track/bdd100k_mot_val_coco.json",
            ),
        },
    }

    # Register data for different domains as well as different sequence.
    domain_path = "bdd100k/labels/box_track_20/domain_splits/"



    train_splits = load_json(
        osp.join("/nobackup-slow/dataset/video/bdd/", domain_path, "bdd100k_mot_domain_splits_train.json")
    )
    val_splits = load_json(
        osp.join("/nobackup-slow/dataset/video/bdd/", domain_path, "bdd100k_mot_domain_splits_val.json")
    )


    # per_seq_{split}_{key}_{_attr}: [dataset_names]
    per_seq_maps = defaultdict(list)

    # register the BDD100K per domain sets
    for split, result in [("train", train_splits), ("val", val_splits)]:
        for key, values in result.items():
            # key is ["timeofday", "scene", "weather"]
            for attr, seqs in values.items():
                # attr is the actual attribute under each category like
                # `daytime`, `night`, etc. Values are list of sequence names.
                if "/" in attr or " " in attr:
                    if "/" in attr:
                        _attr = attr.replace("/", "-")
                    if " " in attr:
                        _attr = attr.replace(" ", "-")
                else:
                    _attr = attr

                # register per domain values.
                for suffix in ["", "_3cls"]:
                    _PREDEFINED_SPLITS_BDDT["bdd_tracking_2k{}".format(suffix)][
                        "bdd_tracking_2k_{}_{}{}".format(split, _attr, suffix)
                    ] = (
                        "bdd100k/images/track/{}".format(split),
                        osp.join(
                            domain_path.replace("box_track_20",
                                                "box_track_20{}".format(suffix)),
                            "labels",
                            split,
                            "{}_{}_{}_coco.json".format(split, key, _attr),
                        ),
                    )


    def register_all_bdd_tracking(root="/nobackup-slow/dataset/video/bdd/"):
        # bdd_tracking meta data
        # fmt: off
        thing_classes = ['pedestrian', 'rider', 'car', 'truck', 'bus',
        'train', 'motorcycle', 'bicycle']
        # thing_classes = ['person', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train']
        thing_classes_3cls = ["vehicle", "pedestrian", "cyclist"]
        # fmt: on
        for DATASETS in [_PREDEFINED_SPLITS_BDDT]:
            for key, value in DATASETS.items():
                metadata = {
                    "thing_classes": thing_classes_3cls
                    if "3cls" in key
                    else thing_classes
                }
                for name, (img_dir, label_file) in value.items():
                    register_coco_instances(
                        name,
                        metadata,
                        os.path.join(root, label_file),
                        os.path.join(root, img_dir),
                    )

        #register the nuscene ood dataset.
        metadata = {  "thing_classes": ['car', 'truck',
                                        'trailer', 'bus',
                                        'construction_vehicle', 'bicycle',
                                        'motorcycle', 'pedestrian',
                                        'traffic_cone', 'barrier', 'animal', 'debris',
                                        'pushable_pullable', 'mobility', 'stroller', 'wheelchair']}
        register_coco_instances(
            'nu_bdd_ood',
            metadata,
            '/nobackup-slow/dataset/video/nuscene/nu_ood.json',
            "/nobackup-slow/dataset/video/nuscene/",
        )

        # register the coco ood dataset wrt the vis dataset.
        metadata = {
            "thing_classes": ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                              'bus', 'train', 'truck', 'boat', 'traffic light',
                              'fire hydrant', 'stop sign', 'parking meter', 'bench',
                              'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                              'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                              'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                              'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                              'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                              'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                              'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                              'hair drier', 'toothbrush']
        }
        register_coco_instances(
            'vis_coco_ood',
            metadata,
            '/nobackup-slow/dataset/coco2017/annotations/instances_val2017_ood_wrt_vis.json',
            "/nobackup-slow/dataset/coco2017/train2017/",
        )



    def register_vis_dataset(root='/nobackup/dataset/video/vis/'):
        thing_classes = ['airplane', 'bear', 'bird', 'boat', 'car', 'cat', 'cow', 'deer', 'dog', 'duck',
                   'earless_seal', 'elephant', 'fish', 'flying_disc', 'fox', 'frog', 'giant_panda',
                   'giraffe', 'horse', 'leopard', 'lizard', 'monkey', 'motorbike', 'mouse', 'parrot',
                   'person', 'rabbit', 'shark', 'skateboard', 'snake', 'snowboard', 'squirrel', 'surfboard',
                   'tennis_racket', 'tiger', 'train', 'truck', 'turtle', 'whale', 'zebra']
        # thing_classes = ['airplane', 'bear', 'car', 'cat', 'deer', 'dog','elephant',  'giant_panda',
        #                      'giraffe', 'horse', 'motorbike',
        #                      'person', 'rabbit', 'skateboard', 'snake',
        #                      'tiger','truck' ]
        metadata = {"thing_classes": thing_classes}
        register_coco_instances(
            'vis21_val',
            metadata,
            '/nobackup-slow/dataset/video/vis/train/instances_test.json',
            "/nobackup-slow/dataset/video/vis/train/JPEGImages/",
        )
        register_coco_instances(
            'vis21_train',
            metadata,
            '/nobackup-slow/dataset/video/vis/train/instances_train.json',
            "/nobackup-slow/dataset/video/vis/train/JPEGImages/",
        )


    # ===== register vanilla coco dataset ====
    from detectron2.data import MetadataCatalog
    def register_all_coco(dataset_dir='/nobackup-slow/dataset/coco2017'):
        thing_classes = MetadataCatalog.get('coco_2017_train').thing_classes
        metadata = {"thing_classes": thing_classes}
        train_json_annotations = os.path.join(
            dataset_dir, 'annotations', 'instances_train2017.json')
        test_json_annotations = os.path.join(
            dataset_dir, 'annotations', 'instances_val2017.json')
        train_image_dir = os.path.join(dataset_dir, 'train2017')
        test_image_dir = os.path.join(dataset_dir, 'val2017')
        register_coco_instances('coco_2017_train_custom', metadata,
                                train_json_annotations,
                                train_image_dir)
        register_coco_instances('coco_2017_val_custom', metadata,
                                test_json_annotations,
                                test_image_dir)

    def register_coco_ood_wrt_bdd(dataset_dir='/nobackup-slow/dataset/coco2017'):
        thing_classes = MetadataCatalog.get('coco_2017_train').thing_classes
        metadata = {"thing_classes": thing_classes}

        test_json_annotations = os.path.join(
            dataset_dir, 'annotations', 'instances_val2017_ood_wrt_bdd.json')
        test_image_dir = os.path.join(dataset_dir, 'val2017')

        register_coco_instances('coco_2017_val_ood_wrt_bdd', metadata,
                                test_json_annotations,
                                test_image_dir)


