import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch import nn
import torchvision.ops as ops
import math

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple, cross_entropy
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads import (
    build_box_head,
    build_keypoint_head,
    build_mask_head,
    FastRCNNOutputLayers,
)
from detectron2.modeling.roi_heads.roi_heads import (
    ROIHeads,
    ROI_HEADS_REGISTRY,
    select_foreground_proposals,
    select_proposals_with_visible_keypoints,
)

from .fast_rcnn import FastRCNNOutputs

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsSS(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        return_roi: bool = False,
        box_roi_thr: Optional[float] = None,
        box_roi_all: Optional[float] = None,
        box2box_transform: Any = None,
        smooth_l1_beta: Any = None,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

        # init for returning ROIs for cyclehead as well as old parameters
        # from detectron2 API changes.
        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta

        self.return_roi = return_roi
        self.box_roi_thr = box_roi_thr
        self.box_roi_all = box_roi_all

    @classmethod
    def from_config(cls, cfg, input_shape):
        cls.cfg = cfg
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        if inspect.ismethod(cls._init_ttt):
            ret.update(cls._init_ttt(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels,
                width=pooler_resolution,
                height=pooler_resolution,
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels,
                width=pooler_resolution,
                height=pooler_resolution,
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        return ret

    @classmethod
    def _init_ttt(cls, cfg, input_shape):
        return_roi = "build_cycle_energy_direct_add_all_head" in cfg.MODEL.SS.NAME
        box_roi_thr = cfg.MODEL.SS.ROI_THR
        if cfg.MODEL.SS.NAME[0] == 'build_cycle_energy_direct_add_att_head':
            #if "vis21" in cfg.DATASETS.TRAIN[0]:
            box_roi_thr = 0.8#cfg.MODEL.SS.ROI_THR
        else:
            if "vis21" in cfg.DATASETS.TRAIN[0]:
                box_roi_thr = 0.8
        box_roi_all = cfg.MODEL.SS.ROI_ALL

        # detectron2 API changes
        box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )
        smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

        ret = {
            "return_roi": return_roi,
            "box_roi_thr": box_roi_thr,
            "box_roi_all": box_roi_all,
            "box2box_transform": box2box_transform,
            "smooth_l1_beta": smooth_l1_beta,
        }
        return ret

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        # pylint: disable=no-member
        del images

        box_features = None
        features_list = [features[f] for f in self.in_features]
        # print(features.keys())
        assert len(features_list) == len(self.in_features), len(features_list)
        for i in range(len(features_list)):
            assert torch.isnan(features_list[i]).sum() == 0, i

        if self.return_roi:
            if self.cfg.MODEL.SS.NAME[0] != 'build_cycle_energy_direct_add_att_head':
                box_features = self._return_box_features(
                    features_list, proposals, targets
                )
            else:
                box_features = self._return_box_features(
                    features_list, proposals, targets
                )
        if self.training and targets is not None:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        for i in range(len(features_list)):
            assert features_list[i].size(0) == len(proposals), features_list[
                i
            ].size(0)
        for i in range(len(proposals)):
            assert len(proposals[i]) > 0, proposals[i]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            # if isinstance(losses, tuple):
            #     losses, box_features = losses
            # During training the proposals used by the box head are
            # used by the mask, keypoint (and densepose) heads.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, (losses, box_features)
        else:
            pred_instances = self._forward_box(features_list, proposals)
            # if isinstance(pred_instances, tuple):
            #     pred_instances, box_features = pred_instances
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(
                features, pred_instances
            )
            return pred_instances, ({}, box_features)

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has(
            "pred_classes"
        )

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances





    def _return_box_features(self, features, proposals, targets=None):
        # pylint: disable=no-member
        proposals2 = []
        idxs = torch.zeros(len(proposals)).int()
        gt_logits = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
        self.box_roi_thr_logits = math.log(
            (1.0 - self.box_roi_thr) / (1.0 - (1.0 - self.box_roi_thr))
        )
        for i, p in enumerate(proposals):
            boxes = p.proposal_boxes.tensor
            scores = p.objectness_logits

            if self.training and targets is not None:
                gt_boxes = targets[i].gt_boxes.tensor
                gt_scores = gt_logits * torch.ones(len(gt_boxes)).to(
                    scores.device
                )
                boxes = torch.cat([gt_boxes, boxes])
                scores = torch.cat([gt_scores, scores])

            if not self.box_roi_all:
                # filter out the low scores
                keep_scores = torch.nonzero(
                    scores > self.box_roi_thr_logits
                ).squeeze(1)
                # print(len(boxes), len(keep_scores), self.cfg.MODEL.SS.ROI_THR)
                boxes = boxes[keep_scores]
                scores = scores[keep_scores]
                # print('num boxes: ', len(boxes))
                if len(boxes.size()) == 1:
                    boxes = boxes.unsqueeze(0)

                # assert boxes.size(0) > 0, targets[i]

                if boxes.size(0) > 5:
                    # keep = batched_nms(boxes, scores, idxs, 0.1)
                    if self.cfg.MODEL.SS.NAME[0] != 'build_cycle_energy_direct_add_att_head' and \
                            self.cfg.MODEL.SS.NAME[0] != "build_cycle_energy_direct_add_att_neg_head":
                        keep = ops.nms(boxes, scores, 0.1)
                    else:
                        keep = ops.nms(boxes, scores, 0.1)

                    if len(keep) > 5:
                        # filtering scores
                        boxes = boxes[keep]
                        scores = scores[keep]
            # print(len(boxes))
            p2 = Instances(p._image_size)
            p2.proposal_boxes = Boxes(boxes)
            p2.objectness_logits = scores
            # print('num_boxes: ', len(boxes))
            proposals2.append(p2)
            prev = idxs[i - 1] if i > 0 else 0
            idxs[i] = len(boxes) + prev

        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals2]
        )
        # box_features = nn.AdaptiveAvgPool2d(1)(box_features)
        # return the corresponding proposals as well
        return box_features, idxs, proposals2

    def _forward_box_additional(self, box_features):
        """
        Forward logic of the box prediction branch.
        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        # box_features = self.box_pooler(
        #     features, [x.proposal_boxes for x in proposals]
        # )
        # box_features_clone = box_features.clone()
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        # print(len(box_features))
        if self.training:
            # breakpoint()
            gt_classes = torch.ones(len(box_features))*(self.cfg.MODEL.ROI_HEADS.NUM_CLASSES-1)
            losses = {'loss_cls_add': cross_entropy(predictions[0], gt_classes.cuda().long(), reduction="mean")}
            # losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.

            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(
                predictions, proposals
            )
            return pred_instances


    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.
        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        # box_features_clone = box_features.clone()
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        # print(len(box_features))
        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = (
                        self.box_predictor.predict_boxes_for_gt_classes(
                            predictions, proposals
                        )
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(
                            pred_boxes_per_image
                        )
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(
                predictions, proposals
            )
            return pred_instances


    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(
                instances, self.num_classes
            )

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [
                x.proposal_boxes if self.training else x.pred_boxes
                for x in instances
            ]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(
                instances, self.num_classes
            )
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [
                x.proposal_boxes if self.training else x.pred_boxes
                for x in instances
            ]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.keypoint_in_features}
        return self.keypoint_head(features, instances)
