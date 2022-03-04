import argparse
import contextlib
import copy
import io
import itertools
import json
import math
import numpy as np
import os.path as osp
from collections import OrderedDict
from tabulate import tabulate
from detectron2.evaluation import print_csv_format
from detectron2.data import MetadataCatalog
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    # Use the COCO default keypoint OKS sigmas unless overrides are specified
    if kpt_oks_sigmas:
        coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def _derive_coco_results(coco_eval, iou_type, class_names):
    """
    Derive the desired score numbers from summarized COCOeval.
    Args:
        coco_eval (None or COCOEval): None represents no predictions from model.
        iou_type (str):
        class_names (None or list[str]): if provided, will use it to predict
            per-category AP.
    Returns:
        a dict of {metric name: score}
    """

    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }[iou_type]

    if coco_eval is None:
        print("No predictions from the model! Set scores to -1")
        return {metric: -1 for metric in metrics}

    # the standard metrics
    results = {
        metric: float(coco_eval.stats[idx] * 100) \
            if float(coco_eval.stats[idx] * 100) != -100 else float("nan") \
                for idx, metric in enumerate(metrics)
    }

    if class_names is None or len(class_names) <= 1:
        return results
    # Compute per-category AP
    # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
    precisions = coco_eval.eval["precision"]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    results_per_category = []
    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        results_per_category.append(("{}".format(name), float(ap * 100)))

    # tabulate it
    N_COLS = min(6, len(results_per_category) * 2)
    results_flatten = list(itertools.chain(*results_per_category))
    results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        results_2d,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=["category", "AP"] * (N_COLS // 2),
        numalign="left",
    )
    print("Per-category {} AP: \n".format(iou_type) + table)

    results.update({"AP-" + name: ap for name, ap in results_per_category})

    return results


def set_coco_api(args, dataset_name, seqs):
    if args.num != -1 and not args.direct_avg:
        # create temp json file if not using all sequences
        save_file = osp.join(
            'datasets/waymo_labels/subsets',
            '{}_{}seqs.json'.format(dataset_name, args.num),
        )
        if not osp.exists(save_file):
            new_imgs = []
            new_annos = []
            for seq in seqs[0]:
                metadata = MetadataCatalog.get(seq)
                json_file = PathManager.get_local_path(metadata.json_file)
                data = json.load(open(json_file))
                new_imgs.extend(data['images'])
                new_annos.extend(data['annotations'])
            new_data = {
                'videos': data['videos'],
                'images': new_imgs,
                'categories': data['categories'],
                'annotations': new_annos,
            }
            if 'cameras' in data:
                new_data['cameras'] = data['cameras']
            with open(save_file, 'w') as fp:
                json.dump(new_data, fp)
        else:
            metadata = MetadataCatalog.get(dataset_name)
            new_data = json.load(open(save_file))
        img_ids = list(set([img['id'] for img in new_data['images']]))
        json_file = save_file
    else:
        metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(metadata.json_file)
        img_ids = None

    if args.direct_avg and args.full_data:
        data = json.load(open(json_file))
        img_ids = list(set([img['id'] for img in data['images']]))

    # set up coco_api
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    sub_apis = []
    img_split = None
    basename = osp.splitext(json_file)[0].replace('_full', '')
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(args.range):
                name = basename + '_{}.json'.format(i)
                api = COCO(name)
                sub_apis.append(api)
        # print(basename+'_img_ids.json')
        with open(basename+'_img_ids.json') as fp:
            img_split = json.load(fp)
    except:
        pass

    # thing_classes
    thing_clases = metadata.get("thing_classes")
    return coco_api, thing_clases, sub_apis, img_split, img_ids


def load_json(filename):
    with open(filename, 'r') as fp:
        out = json.load(fp)
    return out


def dump_json(filename, value):
    with open(filename, 'w') as fp:
        json.dump(fp, value)


def parse_data_seqs(json_path):
    # txt_path = osp.join(args.output_dir, args.seq_txt)
    with open(json_path) as fp:
        res = json.load(fp)
    # with open(txt_path) as fp:
    #     res = [r.strip() for r in fp.readlines()]
    return res


# def split_res_time(res):
#     seen_images = []
#     idx = 0
#     split_res = [[] for _ in range(5)]
#
#     for r in res:
#         img = r['image_id']
#         if len(seen_images) == 50 and img not in seen_images:
#             idx += 1
#             seen_images = [img]
#         elif len(seen_images) < 50 and img not in seen_images:
#             seen_images.append(img)
#
#         if img in seen_images:
#             split_res[idx].append(r)
#
#     return split_res


def load_data_seqs(args, seqs, img_split, img_ids):
    filenames = [osp.join(args.output_dir, s, args.json_name) for s in seqs]
    agg_pred = []
    seg_agg_pred = [[] for _ in range(args.range)]
    mapping = {1: 2, 2: 3, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
    # import pdb; pdb.set_trace()
    for name in filenames:
        res = load_json(name)
        if img_ids is not None:
            res = [r for r in res if r['image_id'] in img_ids]
        agg_pred += res
        for r in res:
            img = str(r['image_id'])
            if img_split is not None:
                idx = img_split[img]
                seg_agg_pred[idx].append(r)
    return agg_pred, seg_agg_pred


def agg_eval(args, task):
    # Hard-coding the datasets for easier input
    suffix = '_full' if args.full else ""

    # encode the overall tasks here
    # dataset name of the overall data registered in the datasets
    # seq_json: json file of a list of per sequence data names
    if task == 'all':
        overall_data = 'webcam_val'
        seq_json = 'datasets/webcam-one-hour/all_seqs{}.json'.format(suffix)
    elif task == 'day2day':
        overall_data = 'webcam_day2day_val'
        seq_json = 'datasets/webcam-one-hour/day2day_seqs{}.json'.format(suffix)
    elif task == 'day2night':
        overall_data = 'webcam_day2night_val'
        seq_json = 'datasets/webcam-one-hour/day2night_seqs{}.json'.format(suffix)
    elif task == 'bdd_val_night':
        overall_data = 'bdd_tracking_2k_val_night'
        seq_json = 'datasets/bdd-tracking-2k/labels/per_seq/val_night_seqs.json'
    elif task == 'bdd_val_daytime':
        overall_data = 'bdd_tracking_2k_val_daytime'
        seq_json = 'datasets/bdd-tracking-2k/labels/per_seq/val_daytime_seqs.json'
    elif task == 'bdd_val_dawn_dusk':
        overall_data = 'bdd_tracking_2k_val_dawn_dusk'
        seq_json = 'datasets/bdd-tracking-2k/labels/per_seq/val_dawn_dusk_seqs.json'
    elif task == 'waymo_front_val':
        overall_data = 'waymo_front_val'
        seq_json = 'datasets/waymo_labels/maps/waymo12_front_val_3cls_register.json'
    elif task == 'waymo_front_left_val':
        overall_data = 'waymo_front_left_val'
        seq_json = 'datasets/waymo_labels/maps/waymo12_front_left_val_3cls_register.json'
    elif task == 'waymo_front_right_val':
        overall_data = 'waymo_front_right_val'
        seq_json = 'datasets/waymo_labels/maps/waymo12_front_right_val_3cls_register.json'
    elif task == 'waymo_side_left_val':
        overall_data = 'waymo_side_left_val'
        seq_json = 'datasets/waymo_labels/maps/waymo12_side_left_val_3cls_register.json'
    elif task == 'waymo_side_right_val':
        overall_data = 'waymo_side_right_val'
        seq_json = 'datasets/waymo_labels/maps/waymo12_side_right_val_3cls_register.json'

    seqs = parse_data_seqs(seq_json)
    if isinstance(seqs, dict):
        seqs = list(seqs.keys())

    if args.num != -1:
        seqs = seqs[:args.num]

    if args.direct_avg:
        datas = seqs
        seqs = [[seq] for seq in seqs]
        seq_names = [overall_data]
    else:
        datas = [overall_data]
        seqs = [seqs]
        seq_names = seqs
        if args.full_data:
            seqs = [[overall_data]]

    if args.reverse:
        for seq in seqs:
            for i in range(len(seq)):
                seq[i] += '_reverse'

    agg_results = []
    agg_results_ranges = []
    res_pastes = []
    keys = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl']
    if args.super_class or 'waymo' in task:
        keys += ['AP-vehicle', 'AP-pedestrian', 'AP-cyclist']
    else:
        keys += ['AP-person', 'AP-rider', 'AP-car', 'AP-bus',
                 'AP-truck', 'AP-bicycle', 'AP-motorcycle', 'AP-train']

    if args.num != -1 and not args.direct_avg:
        coco_api, thing_classes, sub_apis, img_split, img_ids = set_coco_api(args, task, seq_names)
    for data, seq in zip(datas, seqs):
        if args.num == -1 or args.direct_avg:
            coco_api, thing_classes, sub_apis, img_split, img_ids = set_coco_api(args, data, None)
        agg_pred, seg_agg_pred = load_data_seqs(
            args, seq_names if args.full_data else seq, img_split, img_ids)
        coco_eval = _evaluate_predictions_on_coco(coco_api, agg_pred, 'bbox')
        agg_result = _derive_coco_results(coco_eval, 'bbox', thing_classes)
        agg_results.append(agg_result)
        if args.direct_avg:
            if not args.full_data:
                output_file = osp.join(args.output_dir, seq[0], args.out_name)
        else:
            output_file = osp.join(args.output_dir, args.out_name)

        res_paste = []
        agg_results_range = []
        if not args.direct_avg or not args.full_data:
            with open(output_file, 'w') as fp:
                print(agg_result, file=fp)
        for k in keys:
            if k in agg_result:
                res_paste.append('{:.3f}'.format(agg_result[k]))
                agg_results_range.append(agg_result[k])

        if sub_apis and args.eval_range:
            for i in range(args.range):
                print('period {}'.format(i))
                coco_eval = _evaluate_predictions_on_coco(sub_apis[i], seg_agg_pred[i], 'bbox')
                agg_result = _derive_coco_results(coco_eval, 'bbox', thing_classes)
                output_file = osp.join(args.output_dir, 'agg_results_{}.json'.format(i))
                with open(output_file, 'w') as fp:
                    print(agg_result, file=fp)
                for k in keys:
                    if k in agg_result:
                        res_paste.append('{:.3f}'.format(agg_result[k]))
                        agg_results_range.append(agg_result[k])

        print('easy pasting results:')
        print(','.join(keys))
        print(','.join(res_paste))
        res_pastes.append(res_paste)
        agg_results_ranges.append(agg_results_range)

    print('')
    for data, res_paste in zip(datas, res_pastes):
        print(data)
        print(','.join(keys))
        print(','.join(res_paste))

    if args.direct_avg:
        # Averaging over all sequence APs
        agg_result_final = {k: 0.0 for k in agg_results[0].keys()}
        class_counts = {k: 0 for k in agg_results[0].keys()}
        for agg_res in agg_results:
            for k in agg_res.keys():
                if not math.isnan(agg_res[k]):
                    agg_result_final[k] += agg_res[k]
                    class_counts[k] += 1
        for k in agg_result_final.keys():
            if class_counts[k] != 0:
                agg_result_final[k] /= class_counts[k]
            else:
                agg_result_final[k] = float('nan')
        output_file = osp.join(args.output_dir, args.out_name)
        res_paste = []
        for val in agg_result_final.values():
            res_paste.append('{:.3f}'.format(val))
        res_paste = ','.join(res_paste)
        print('Average over all sequences')
        print(res_paste)
        print('')
        with open(output_file, 'w') as fp:
            print(agg_result_final, file=fp)
        print(agg_result_final)
        print('')

        return res_paste, agg_result_final
    return None, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-o', default='', type=str,  help='root_path')
    # parser.add_argument('--overall-data', '-d', default='webcam_val',
    #                     type=str, help='overall data name')
    # parser.add_argument('--seq-json', '-s', default='', type=str,
    #                     help='txt files for the testing sequences')
    parser.add_argument('--task', '-t', default='bdd_val_night', type=str,
                        # choices=['all', 'day2day', 'day2night', 'bdd_val_night',
                        #          'bdd_val_daytime', 'bdd_val_dawn_dusk',
                        #          'waymo_front_val', 'waymo_front_left_val',
                        #          'waymo_front_right_val', 'waymo_side_left_val',
                        #          'waymo_side_right_val'],
                        help='evaluation task')
    parser.add_argument('--json-name', default='coco_instances_results.json')
    parser.add_argument('--out-name', default='agg_res.json',
                        help='save the agg results')
    parser.add_argument('--step', default=20, type=int,
                        help='step size of frames')
    parser.add_argument('--range', '-r', default=6, type=int,
                        help='range of labels')
    # parser.add_argument('--per-seq', action='store_true',
    #                     help='whether to use per sequence evaluation')
    parser.add_argument('--direct-avg', action='store_true',
                        help='whether to use direct average for aggregation;'
                             'the default is the old aggregation way')
    parser.add_argument('--reverse', action='store_true',
                        help='whether using reverse datasets')
    parser.add_argument('--full', action='store_true',
                        help='recall the full data')
    parser.add_argument('--eval-range', '-e', action='store_true',
                        help='make evaluation of each time range as optional')
    parser.add_argument('--super-class', action='store_true',
                        help='use 3 super classes')
    parser.add_argument('--num', '-n', default=-1, type=int,
                        help='number of sequences to aggregate')
    parser.add_argument('--full-data', action='store_true',
                        help='use full dataset instead of sequences')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    tasks = args.task.split(',')

    res_pastes = []
    for task in tasks:
        res_paste, agg_results = agg_eval(args, task)
        res_pastes.append(res_paste)

    print('')
    print('### All results ###')
    for i in range(len(tasks)):
        print(tasks[i])
        print(res_pastes[i])
        print('')
