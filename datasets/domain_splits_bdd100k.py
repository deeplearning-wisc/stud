"""Label preprocessing.

- Converting bdd100K detection set into per domain labels in COCO format.
- Obtain BDD100K {attr: video_name} mapping.
- Regroup BDD100K tracking videos into different sub domains.
- Convert the sub domain video labels into per-sequence coco labels and
per-domain labels.

Run `python3 -m datasets.domain_splits_bdd100k`
"""

import glob
import json
import os
import os.path as osp
import shutil
import argparse
from collections import defaultdict
from typing import Any, Dict, List

from datasets.bdd100k2coco import bdd100k2coco_det, process

DictObject = Dict[str, Any]


def load_json(filename: str) -> Any:
    """Load json."""
    with open(filename, "r") as fp:
        res = json.load(fp)
    return res


def dump_json(filename: str, value: Any) -> None:
    """Dump data into json files."""
    with open(filename, "w") as fp:
        json.dump(value, fp)


def domain_splits(
    bdd100k_labels: List[DictObject], root: str, split: str = "val"
) -> DictObject:
    """Split bdd100K det set into different sub domains.

    Args:
        bdd100k_labels (List[DictObject]): in BDD format.
        root (str): directory to save the labels.
        split (str, optional): [train/val/test]. Defaults to "val".

    Returns:
        DictObject: Return maps between attributes and video names.
    """
    # key:value = attribute: annotations in BDD format
    weather_dict: DictObject = defaultdict(list)
    scene_dict: DictObject = defaultdict(list)
    timeofday_dict: DictObject = defaultdict(list)

    # key:value = attribute:video_name
    weather_names: DictObject = defaultdict(list)
    scene_names: DictObject = defaultdict(list)
    timeofday_names: DictObject = defaultdict(list)

    for _, item in enumerate(bdd100k_labels):
        attr = item["attributes"]
        weather = attr["weather"]
        scene = attr["scene"]
        timeofday = attr["timeofday"]
        weather_dict[weather] += [item]
        scene_dict[scene] += [item]
        timeofday_dict[timeofday] += [item]

        # for future video spliting as there is no attr tags in videos.
        video_name = item["name"][:-4]
        weather_names[weather] += [video_name]
        scene_names[scene] += [video_name]
        timeofday_names[timeofday] += [video_name]

    save_path = osp.join(root, "domain_splits")
    os.makedirs(save_path, exist_ok=True)
    res = {
        "weather": weather_dict,
        "scene": scene_dict,
        "timeofday": timeofday_dict,
    }
    maps = {
        "weather": weather_names,
        "scene": weather_names,
        "timeofday": timeofday_names,
    }

    # save per domain labels in COCO format.
    for key, value in res.items():
        for k, v in value.items():
            if "/" in k:
                k = k.replace("/", "-")
            if " " in k:
                k = k.replace(" ", "-")
            bdd_ann_file = osp.join(
                save_path, "bdd100k_ann_{}_{}_{}.json".format(split, key, k)
            )
            if not osp.exists(bdd_ann_file):
                dump_json(bdd_ann_file, v)

            coco_ann_file = osp.join(
                save_path,
                "bdd100k_ann_{}_{}_{}_coco.json".format(split, key, k),
            )
            if not osp.exists(coco_ann_file):
                # do data format convert
                coco_anns = bdd100k2coco_det(v)
                dump_json(coco_ann_file, coco_anns)

    # save per domain video splits.
    for key, value in maps.items():
        dump_json(
            osp.join(save_path, "bdd100k_{}_{}.json".format(split, key)), value
        )
    return maps


def convert_videos(
    bdd100k_maps: DictObject, video_root: str, split: str
) -> None:
    """Copy, filter and convert BDD video labels.

    Args:
        bdd100k_maps (DictObject): [description]
        video_root (str): [description]
        split (str): [description]
    """
    video_paths = glob.glob(osp.join(video_root, split, "*.json"))
    video_names = {
        osp.splitext(osp.basename(video))[0]: video for video in video_paths
    }
    save_root = osp.join(video_root, "domain_splits")
    os.makedirs(save_root, exist_ok=True)
    label_path = osp.join(save_root, "labels")
    os.makedirs(label_path, exist_ok=True)

    # attrs: DictObject = {
    #     "timeofday": defaultdict(list),
    #     "scene": defaultdict(list),
    #     "weather": defaultdict(list),
    # }
    # split the tracking sequences into various time of day.
    attrs: DictObject = {
        "timeofday": defaultdict(list)
    }

    for key, value in attrs.items():
        video_maps = bdd100k_maps[key]
        for v, vpath in video_names.items():
            for attr, vids in video_maps.items():
                # process each specific domain.
                if "/" in attr or " " in attr:
                    if "/" in attr:
                        _attr = attr.replace("/", "-")
                    if " " in attr:
                        _attr = attr.replace(" ", "-")
                else:
                    _attr = attr
                pq_save_path = osp.join(
                    video_root,
                    "per_seq",
                    split,
                    "{}_{}_bdd".format(key, _attr),
                )
                coco_save_path = osp.join(
                    video_root,
                    "per_seq",
                    split,
                    "{}_{}_coco".format(key, _attr),
                )
                os.makedirs(pq_save_path, exist_ok=True)
                if v in vids:
                    # add to mapping
                    value[attr] += [v]
                    # copy annotation files
                    if not osp.exists(osp.join(pq_save_path, v + ".json")):
                        shutil.copy(vpath, pq_save_path)
                    # process label conversion.
                    save_res = osp.join(coco_save_path, v + ".json")
                    if not osp.exists(save_res):
                        process(
                            vpath,
                            save_res,
                            "track",
                            remove_ignore=True if split == "train" else False,
                            ignore_as_class=False,
                        )

    # collected all the per domain videos.
    for key in attrs:
        for attr in bdd100k_maps[key]:
            if "/" in attr or " " in attr:
                if "/" in attr:
                    _attr = attr.replace("/", "-")
                if " " in attr:
                    _attr = attr.replace(" ", "-")
            else:
                _attr = attr
            pq_save_path = osp.join(
                video_root,
                "per_seq",
                split,
                "{}_{}_bdd".format(key, _attr),
            )
            process(
                pq_save_path,
                osp.join(
                    label_path,
                    split,
                    "{}_{}_{}_coco.json".format(split, key, _attr),
                ),
                "track",
                remove_ignore=True if split == "train" else False,
                ignore_as_class=False,
            )
    # save mapping dictionary
    dump_json(
        osp.join(save_root, "bdd100k_mot_domain_splits_{}.json".format(split)),
        attrs,
    )

    print("summary:")
    for attr, values in attrs.items():
        print("{}:".format(attr))
        name_str = []
        len_str = []
        for k, v in values.items():
            name_str.append(k)
            len_str.append(str(len(v)))
        print("\t".join(name_str))
        print("\t".join(len_str))


def main() -> None:
    """Main function."""
    bdd100k_path = "/nobackup/dataset/my_xfdu/video/bdd/bdd100k/labels"
    # These labels should be in BDD100K format.
    bdd100k_val_labels = load_json(
        osp.join(bdd100k_path, "det_20", "det_val.json")
    )
    bdd100k_train_labels = load_json(
        osp.join(bdd100k_path, "det_20", "det_train.json")
    )

    print("process validation set")
    val_maps = domain_splits(bdd100k_val_labels, bdd100k_path, split="val")

    print("process train set")
    train_maps = domain_splits(
        bdd100k_train_labels, bdd100k_path, split="train"
    )

    # process video data
    video_path = "/nobackup/dataset/my_xfdu/video/bdd/bdd100k/labels/box_track_20"
    convert_videos(val_maps, video_path, "val")
    convert_videos(train_maps, video_path, "train")

    print("done")


if __name__ == "__main__":
    main()
