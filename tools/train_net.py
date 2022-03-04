"""Train/eval script."""
import logging
import os
import os.path as osp
import time
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import default_setup, hooks, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)

# updated code
from src.config import get_cfg
from src import data
from src.engine import default_argument_parser, DefaultTrainer
from src import modeling


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    # eval_only and eval_during_train are mainly used for jointly
    # training detection and self-supervised models.
    # breakpoint()
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        # breakpoint()
        position_list = [x for x, v in enumerate(cfg.MODEL.WEIGHTS) if v == '/']
        if 'ood' not in cfg.DATASETS.TEST[0]:
            res = Trainer.test(cfg, model,
                               saved_address=cfg.MODEL.WEIGHTS[:position_list[-1]] + '/id.npy',
                               visualize=args.visualize, savefigdir=args.savefigdir)
            if comm.is_main_process():
                verify_results(cfg, res)
            if cfg.TEST.AUG.ENABLED:
                res.update(Trainer.test_with_TTA(cfg, model))
            return res
        else:
            res = Trainer.test(cfg, model,
                               saved_address=cfg.MODEL.WEIGHTS[:position_list[-1]] + '/ood.npy',
                               visualize=args.visualize, savefigdir=args.savefigdir)
            return res

    elif args.eval_during_train:#False
        model = Trainer.build_model(cfg)
        check_pointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        saved_checkpoint = None
        best_res = {}
        best_file = None
        while True:
            if check_pointer.has_checkpoint():
                current_ckpt = check_pointer.get_checkpoint_file()
                if (
                    saved_checkpoint is None
                    or current_ckpt != saved_checkpoint
                ):
                    check_pointer._load_model(
                        check_pointer._load_file(current_ckpt)
                    )
                    saved_checkpoint = current_ckpt
                    print("evaluating checkpoint {}".format(current_ckpt))
                    iters = int(
                        osp.splitext(osp.basename(current_ckpt))[0].split("_")[
                            -1
                        ]
                    )
                    res = Trainer.test(cfg, model)
                    if comm.is_main_process():
                        verify_results(cfg, res)
                    if cfg.TEST.AUG.ENABLED:
                        res.update(Trainer.test_with_TTA(cfg, model))
                    print(res)
                    if (len(best_res) == 0) or (
                        len(best_res) > 0
                        and best_res["bbox"]["AP"] < res["bbox"]["AP"]
                    ):
                        best_res = res
                        best_file = current_ckpt
                    print("best so far is from {}".format(best_file))
                    print(best_res)
                    if iters + 1 >= cfg.SOLVER.MAX_ITER:
                        return best_res
            time.sleep(10)
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [
                hooks.EvalHook(
                    0, lambda: trainer.test_with_TTA(cfg, trainer.model)
                )
            ]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
