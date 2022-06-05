import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import batched_nms
from detectron2.structures import BoxMode, Boxes, Instances
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator

# from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n

import logging
import math

from ..self_supervised import build_ss_head
from ..roi_heads import build_roi_heads

__all__ = ["SSRCNNAdd"]


@META_ARCH_REGISTRY.register()
class SSRCNNAdd(nn.Module):
    """
    Detection + self-supervised
    """

    def __init__(self, cfg):
        super().__init__()
        # pylint: disable=no-member
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape()
        )
        self.from_config(cfg)
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.ss_head = build_ss_head(
            cfg, self.backbone.bottom_up.output_shape()
        )

        for i in range(len(self.ss_head)):
            setattr(self, "ss_head_{}".format(i), self.ss_head[i])
        # breakpoint()
        self.to(self.device)
        self.cfg = cfg
        if 'energy' in self.cfg.MODEL.SS.NAME[0]:
            self.logistic_regression = torch.nn.Linear(1, 2)
            # self.logistic_regression = torch.nn.Linear(1, 1)
            self.logistic_regression.cuda()
        index = 0
        self.save_id = 0
        # if self.cfg.MODEL.SS.LOSS == 'additional':
        #     self.add_head = torch.nn.Linear(1024,self.cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1)
        if self.cfg.MODEL.FREEZE == 1:
            for param in self.backbone.bottom_up.parameters():
                # print(param)
                # breakpoint()
                index += 1
                param.requires_grad = False
            # self.backbone.requires_grad = False
            # breakpoint()

    def from_config(self, cfg):
        # only train/eval the ss branch for debugging.
        self.ss_only = cfg.MODEL.SS.ONLY
        self.feat_level = cfg.MODEL.SS.FEAT_LEVEL  # res4

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        pixel_std = (
            torch.Tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(num_channels, 1, 1)
        )
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def forward(self, batched_inputs):
        """
        Training methods, which jointly train the detector and the
        self-supervised task.
        """
        if not self.training:

            return self.inference_my(batched_inputs)
        losses = {}
        accuracies = {}
        # torch.save(batched_inputs, "inputs.pt")
        for i in range(len(self.ss_head)):
            """Using images as the input to SS tasks."""
            head = getattr(self, "ss_head_{}".format(i))
            if head.input != "images":
                continue
            out, tar, ss_losses = head(
                batched_inputs, self.backbone.bottom_up, self.feat_level
            )  # attach new parameters
            losses.update(ss_losses)
            acc = (out.argmax(axis=1) == tar).float().mean().item() * 100
            accuracies["accuracy_ss_{}".format(head.name)] = {
                "accuracy": acc,
                "num": len(tar),
            }

        # for detection part
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None
        # print(images.tensor.size(), images.image_sizes)
        features = self.backbone(images.tensor)
        # print(features['p2'].size(),features['p3'].size(), features['p4'].size(), features['p5'].size(), features['p6'].size())
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [
                x["proposals"].to(self.device) for x in batched_inputs
            ]
            proposal_losses = {}
        # print(len(proposals), proposals[0])

        _, detector_losses, add_stuff = self.roi_heads(
            images, features, proposals, gt_instances
        )
        # breakpoint()

        if isinstance(detector_losses, tuple):
            # breakpoint()
            detector_losses, box_features = detector_losses
            # box features that contain frames in intervals.

            for i in range(len(self.ss_head)):
                head = getattr(self, "ss_head_{}".format(i))
                if head.input != "ROI":
                    continue
                # during training, the paired of inputs are put in one batch
                # breakpoint()
                criterion = torch.nn.CrossEntropyLoss()
                if 'build_cycle_energy_head' == self.cfg.MODEL.SS.NAME[0] or \
                        'build_cycle_energy_direct_head' == self.cfg.MODEL.SS.NAME[0] or  \
                        "build_cycle_energy_direct_add_head" == self.cfg.MODEL.SS.NAME[0] or \
                        "build_cycle_energy_direct_add_max_head" == self.cfg.MODEL.SS.NAME[0] or \
                        "build_cycle_energy_direct_add_random_head" == self.cfg.MODEL.SS.NAME[0] or \
                        "build_cycle_energy_direct_no_head" == self.cfg.MODEL.SS.NAME[0] or \
                        "build_cycle_energy_direct_add_all_head" == self.cfg.MODEL.SS.NAME[0] or \
                        "build_cycle_energy_direct_add_all_max_head" == self.cfg.MODEL.SS.NAME[0] or \
                    "build_cycle_energy_direct_add_all_random_head" == self.cfg.MODEL.SS.NAME[0] or \
                    "build_cycle_energy_direct_add_all_mild_head" == self.cfg.MODEL.SS.NAME[0] or \
                    "build_cycle_energy_direct_add_all_noise_head" == self.cfg.MODEL.SS.NAME[0] or \
                        "build_cycle_energy_direct_add_att_neg_head" == self.cfg.MODEL.SS.NAME[0]:
                    
                    ss_losses, acc, fea_for_reg, fake_loss = head(self.roi_heads,
                                                                                    box_features)
                    if fea_for_reg is None:
                        print('fea_u contains no features!')
                        # ene_loss = torch.zeros(1).cuda()
                        # ene_loss.requires_grad = True
                        fake_input = torch.zeros(1).cuda()
                        fake_loss_for_lr = (self.logistic_regression(fake_input) - \
                                            self.logistic_regression(fake_input)).sum()
                        losses.update({'ene_reg_loss': fake_loss + fake_loss_for_lr})
                    else:
                        if self.cfg.MODEL.SS.LOSS == 'normal':
                            fea_for_reg = self.roi_heads.box_head(fea_for_reg.view(-1, 256, 7, 7))
                            predictions = self.roi_heads.box_predictor(fea_for_reg)
                            binary_labels = torch.ones(len(predictions[0])).cuda()
                            binary_labels[int(len(predictions[0]) / 2):] = 0
                            energy_reg_loss = criterion(self.logistic_regression(
                                torch.logsumexp(predictions[0][:, :-1], dim=1).unsqueeze(1)),
                                binary_labels.long())
                            ene_loss = self.cfg.MODEL.SS.ENERGY_WEIGHT * energy_reg_loss
                        elif self.cfg.MODEL.SS.LOSS == 'margin':
                            fea_for_reg = self.roi_heads.box_head(fea_for_reg.view(-1, 256, 7, 7))
                            predictions = self.roi_heads.box_predictor(fea_for_reg)
                            binary_labels = torch.ones(len(predictions[0])).cuda()
                            binary_labels[int(len(predictions[0]) / 2):] = 0
                            Ec_out = -torch.logsumexp(predictions[0][:, :-1], dim=1)[int(len(predictions[0]) / 2):]
                            Ec_in = -torch.logsumexp(predictions[0][:, :-1], dim=1)[:int(len(predictions[0]) / 2)]
                            ene_loss = 0.001 * (torch.pow(F.relu(Ec_in + 15), 2).mean() + torch.pow(
                                F.relu(-7 - Ec_out), 2).mean())
                            fake_input = torch.zeros(1).cuda()
                            ene_loss1 = (self.logistic_regression(fake_input) - \
                                         self.logistic_regression(fake_input)).sum()
                            ene_loss += ene_loss1
                        elif self.cfg.MODEL.SS.LOSS == 'additional':
                            # print()
                            ene_loss = self.roi_heads._forward_box_additional(
                                fea_for_reg[int(len(fea_for_reg)/2):].view(-1, 256, 7, 7), add_stuff)
                            binary_labels = torch.ones(1).cuda()
                            fake_input = torch.zeros(1).cuda()
                            ene_loss1 = (self.logistic_regression(fake_input) - \
                                                self.logistic_regression(fake_input)).sum()
                            ene_loss = ene_loss['loss_cls_add'] + ene_loss1


                        losses.update({'ene_reg_loss': ene_loss})
                        del binary_labels
                elif "build_cycle_energy_direct_add_att_head" == self.cfg.MODEL.SS.NAME[0]:

                    # breakpoint()
                    # print(box_features[0].shape)
                    ss_losses, acc, fea_for_reg, fake_loss = head(box_features)
                    if fea_for_reg is None:
                        print('fea_u contains no features!')
                        # ene_loss = torch.zeros(1).cuda()
                        # ene_loss.requires_grad = True
                        fake_input = torch.zeros(1).cuda()
                        fake_loss_for_lr = (self.logistic_regression(fake_input) - \
                                            self.logistic_regression(fake_input)).sum()
                        losses.update({'ene_reg_loss': fake_loss + fake_loss_for_lr})
                    else:
                        fea_for_reg = self.roi_heads.box_head(fea_for_reg.view(-1, 256, 7, 7))
                        predictions = self.roi_heads.box_predictor(fea_for_reg)
                        binary_labels = torch.ones(len(predictions[0])).cuda()
                        binary_labels[int(len(predictions[0]) / (self.cfg.DATALOADER.SELCTED_NUMBER + 1)):] = 0

                        # breakpoint()
                        energy_reg_loss = criterion(self.logistic_regression(
                            torch.logsumexp(predictions[0][:, :-1], dim=1).unsqueeze(1)),
                            binary_labels.long())
                        ene_loss = self.cfg.MODEL.SS.ENERGY_WEIGHT * energy_reg_loss
                        # ene_loss.requires_grad=True
                        losses.update({'ene_reg_loss': ene_loss})
                        # print('hhh')
                        del binary_labels

                elif 'build_cycle_energy_1024_latter_head' == self.cfg.MODEL.SS.NAME[0]:
                    ss_losses, acc, fea_for_reg = head(box_features)
                    if fea_for_reg is None:
                        print('fea_u contains no features!')
                        losses.update({'ene_reg_loss': torch.zeros(1).cuda()})
                    else:
                        # breakpoint()
                        # fea_for_reg = self.roi_heads.box_head(fea_for_reg.view(-1, 256, 7, 7))
                        fea_for_reg = self.roi_heads.box_head.fc2(fea_for_reg)
                        fea_for_reg = self.roi_heads.box_head.fc_relu2(fea_for_reg)

                        predictions = self.roi_heads.box_predictor(fea_for_reg)
                        binary_labels = torch.ones(len(predictions[0])).cuda()
                        binary_labels[int(len(predictions[0]) / 2):] = 0
                        # criterion = torch.nn.CrossEntropyLoss()
                        # breakpoint()
                        energy_reg_loss = criterion(self.logistic_regression(
                            torch.logsumexp(predictions[0][:, :-1], dim=1).unsqueeze(1)),
                            binary_labels.long())
                        losses.update({'ene_reg_loss': self.cfg.MODEL.SS.ENERGY_WEIGHT * energy_reg_loss})
                else:
                    ss_losses, acc = head(box_features)
                if "energy" not in self.cfg.MODEL.SS.NAME[0]:
                    losses.update(ss_losses)
                accuracies["accuracy_ss_{}".format(head.name)] = {
                    "accuracy": acc,
                    "num": 1,
                }

        losses.update(detector_losses)
        losses.update(proposal_losses)

        for k, v in losses.items():
            assert math.isnan(v) == False, k

        return losses

    def det_inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [
                    x["proposals"].to(self.device) for x in batched_inputs
                ]

            results, others = self.roi_heads(images, features, proposals, None)
            if isinstance(others, tuple):
                others, box_features = others
                # do not need box features in the inference stage.

            else:
                box_features = None
        else:
            detected_instances = [
                x.to(self.device) for x in detected_instances
            ]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )
            box_features = None

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            # breakpoint()
            return processed_results, box_features
        else:
            return results, box_features

    def inference_my(self,
                     batched_inputs,
                     detected_instances=None,
                     do_postprocess=True,
                     ):
        assert not self.training

        raw_output = dict()

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, _ = self.proposal_generator(images, features, None)
        # Create raw output dictionary
        raw_output.update({'proposals': proposals[0]})
        # results, _ = self.model.roi_heads(images, features, proposals, None)

        features = [features[f] for f in self.roi_heads.box_in_features]
        box_features = self.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.roi_heads.box_head(box_features)
        predictions = self.roi_heads.box_predictor(box_features)
        # import ipdb;
        # ipdb.set_trace()


        box_cls = predictions[0]
        box_delta = predictions[1]
        raw_output.update({'box_cls': box_cls,
                           'box_delta': box_delta})
        outputs = raw_output

        proposals = outputs['proposals']
        box_cls = outputs['box_cls']
        box_delta = outputs['box_delta']


        inter_feat = torch.nn.functional.softmax(box_cls, dim=1)[:, -2]
        box_cls = torch.nn.functional.softmax(box_cls, dim=-1)

        # Remove background category
        scores = box_cls[:, :-1]

        num_bbox_reg_classes = box_delta.shape[1] // 4
        box_delta = box_delta.reshape(-1, 4)
        box_delta = box_delta.view(-1, num_bbox_reg_classes, 4)
        filter_mask = scores > self.roi_heads.box_predictor.test_score_thresh#self.test_score_thres

        filter_inds = filter_mask.nonzero(as_tuple=False)
        # breakpoint()
        if num_bbox_reg_classes == 1:
            box_delta = box_delta[filter_inds[:, 0], 0]
        else:
            box_delta = box_delta[filter_mask]

        det_labels = torch.arange(scores.shape[1], dtype=torch.long)
        det_labels = det_labels.view(1, -1).expand_as(scores)

        scores = scores[filter_mask]
        det_labels = det_labels[filter_mask]

        inter_feat = inter_feat[filter_inds[:, 0]]
        # breakpoint()
        proposal_boxes = proposals.proposal_boxes.tensor[filter_inds[:, 0]]
        # breakpoint()

        # predict boxes
        boxes = self.roi_heads.box_predictor.box2box_transform.apply_deltas(
            box_delta, proposal_boxes)

        outputs = (boxes, scores, inter_feat, filter_inds[:,
                                                        1], box_cls[filter_inds[:, 0]], det_labels)
        results = self.general_standard_nms_postprocessing(batched_inputs, outputs,
                                                 self.roi_heads.box_predictor.test_nms_thresh,
                                                 self.roi_heads.box_predictor.test_topk_per_image)
        # print(len(batched_inputs))
        output_height = batched_inputs[0].get("height", results.image_size[0])
        output_width = batched_inputs[0].get("width", results.image_size[1])

        scale_x, scale_y = (output_width /
                            results.image_size[1], output_height /
                            results.image_size[0])
        results = Instances((output_height, output_width), **results.get_fields())

        output_boxes = results.pred_boxes

        # Scale bounding boxes
        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)
        results = results[output_boxes.nonempty()]

        return [{'instances':results}]

        # breakpoint()


    def general_standard_nms_postprocessing(self, input_im,
                                            outputs,
                                            nms_threshold=0.5,
                                            max_detections_per_image=100):
        """

        Args:
            input_im (list): an input im list generated from dataset handler.
            outputs (list): output list form model specific inference function
            nms_threshold (float): non-maximum suppression threshold
            max_detections_per_image (int): maximum allowed number of detections per image.

        Returns:
            result (Instances): final results after nms

        """
        logistic_score = None
        try:
            predicted_boxes, predicted_prob, inter_feat, \
            classes_idxs, predicted_prob_vectors, det_labels = outputs
        except:
            predicted_boxes, predicted_prob, inter_feat, logistic_score, \
            classes_idxs, predicted_prob_vectors, det_labels = outputs

        # Perform nms
        keep = batched_nms(
            predicted_boxes,
            predicted_prob,
            classes_idxs,
            nms_threshold)
        keep = keep[: max_detections_per_image]
        # import ipdb; ipdb.set_trace()
        # Keep highest scoring results
        result = Instances(
            (input_im[0]['image'].shape[1],
             input_im[0]['image'].shape[2]))
        result.pred_boxes = Boxes(predicted_boxes[keep])
        result.scores = predicted_prob[keep]
        result.pred_classes = classes_idxs[keep]
        result.pred_cls_probs = predicted_prob_vectors[keep]
        result.inter_feat = inter_feat[keep]
        result.det_labels = det_labels[keep]
        if logistic_score is not None:
            result.logistic_score = logistic_score[keep]


        return result


    def inference(
        self, batched_inputs, detected_instances=None, do_postprocess=True
    ):
        """ used for standard detectron2 test method"""
        results, _ = self.det_inference(
            batched_inputs, detected_instances, do_postprocess
        )
        return results

    def preprocess_image(self, batched_inputs):
        """normalize, pad and batch the input images"""
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility
        )
        return images
