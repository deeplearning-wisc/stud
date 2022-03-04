"""BDD .
- Converting BDD labels to 3 classes to match the Waymo dataset.

Run `python3 -m datasets.convert_bdd_3cls`
"""

import json
import os
import os.path as osp
from collections import defaultdict


def load_json(filename):
    with open(filename, "r") as fp:
        reg_file = json.load(fp)
    return reg_file


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
    osp.join("/nobackup-slow/dataset/my_xfdu/video/bdd/", domain_path, "bdd100k_mot_domain_splits_train.json")
)
val_splits = load_json(
    osp.join("/nobackup-slow/dataset/my_xfdu/video/bdd/", domain_path, "bdd100k_mot_domain_splits_val.json")
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
            _PREDEFINED_SPLITS_BDDT["bdd_tracking_2k"][
                "bdd_tracking_2k_{}_{}".format(split, _attr)
            ] = (
                "bdd100k/images/track/{}".format(split),
                osp.join(
                    domain_path,
                    "labels",
                    split,
                    "{}_{}_{}_coco.json".format(split, key, _attr),
                ),
            )

MAPPING = {1: 2, 2: 3, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
os.makedirs(osp.join('/nobackup-slow/dataset/my_xfdu/video/bdd/', 'bdd100k/labels/track_3cls'), exist_ok=True)
os.makedirs(
    osp.join('/nobackup-slow/dataset/my_xfdu/video/bdd/', 'bdd100k/labels/box_track_20_3cls'), exist_ok=True)

datasets = _PREDEFINED_SPLITS_BDDT['bdd_tracking_2k']
files = [datasets[k][1] for k in datasets.keys()]

for json_file in files:
    print(json_file)
    data_path = osp.join('/nobackup-slow/dataset/my_xfdu/video/bdd/', json_file)
    prefix = json_file.split('/')[-1]
    data = json.load(open(data_path))
    new_cats = [
        {'supercategory': 'none', 'id': 1, 'name': 'vehicle'},
        {'supercategory': 'none', 'id': 2, 'name': 'pedestrian'},
        {'supercategory': 'none', 'id': 3, 'name': 'cyclist'},
    ]
    new_annos = []
    for anno in data['annotations']:
        anno['category_id'] = MAPPING[anno['category_id']]
        new_annos.append(anno)
    new_labels = {
        'categories': new_cats,
        'images': data['images'],
        'annotations': new_annos,
        'videos': data['videos'],
    }
    if 'box_track_20' in data_path:
        save_path = osp.dirname(data_path).replace(
            'box_track_20', 'box_track_20_3cls')
    elif 'track' in data_path:
        save_path = osp.dirname(data_path).replace(
            'track', 'track_3cls')
    else:
        assert False
    os.makedirs(save_path, exist_ok=True)
    save_path = osp.join(save_path, prefix)
    with open(save_path, 'w') as fp:
        json.dump(new_labels, fp)
