# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from detectron2.layers import batched_nms
from detectron2.structures import BoxMode, Boxes, Instances
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import Instances
from .Imagelist import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
# from detectron2.modeling.roi_heads import build_roi_heads
# from probabilistic_modeling.roihead_logistic import build_roi_heads
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from .GAN import Generator, Discriminator
from torch.autograd import Variable
import copy

__all__ = ["GeneralizedRCNNLogisticGAN"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNLogisticGAN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.nz = 100
        self.G = Generator(1, self.nz, 64, 3)
        self.D = Discriminator(1, 3, 64)
        self.fixed_noise = Variable(torch.FloatTensor(64, self.nz, 1, 1).normal_(0, 1).cuda())
        self.optimizerD = torch.optim.Adam(self.D.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.optimizerG = torch.optim.Adam(self.G.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.real_label = 1
        self.fake_label = 0
        self.criterion = nn.BCELoss()
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference_my(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        gan_target = torch.FloatTensor(len(batched_inputs)).fill_(0).cuda()
        gan_target.fill_(self.real_label)
        targetv = Variable(gan_target)
        self.optimizerD.zero_grad()
        for index in range(len(batched_inputs)):
            if index == 0:
                data = torch.nn.functional.interpolate(
                    batched_inputs[index]['image'].unsqueeze(0), [32, 32], mode='nearest').float() / 255.
            else:
                data = torch.cat((data, torch.nn.functional.interpolate(
                    batched_inputs[index]['image'].unsqueeze(0), [32, 32], mode='nearest').float() / 255.), dim=0)
        # breakpoint()
        output = self.D(data.to(self.device))
        # breakpoint()
        errD_real = self.criterion(output.squeeze(), targetv)
        errD_real.backward()

        noise = torch.FloatTensor(data.size(0), self.nz, 1, 1).normal_(0, 1).cuda()
        noise = Variable(noise)
        fake = self.G(noise)
        targetv = Variable(gan_target.fill_(self.fake_label))
        output = self.D(fake.detach())
        # breakpoint()
        errD_fake = self.criterion(output.squeeze(), targetv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        self.optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        self.optimizerG.zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(self.real_label))
        output = self.D(fake)
        errG = self.criterion(output.squeeze(), targetv)
        D_G_z2 = output.data.mean()

        # minimize the true distribution
        fake_upsampled_copied = copy.deepcopy(batched_inputs)

        for index in range(len(fake)):
            fake_upsampled_copied[index]['image'] = torch.nn.functional.interpolate(
                fake[index].unsqueeze(0), list(batched_inputs[index]['image'].size())[1:], mode='bilinear') * 255
            fake_upsampled_copied[index]['image'] = fake_upsampled_copied[index]['image'].squeeze()

        fake_images = self.preprocess_image(fake_upsampled_copied)



        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features_fake = self.backbone(fake_images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)

        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        # breakpoint()
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        _, fake_logits = self.roi_heads.forward_special(fake_images, features_fake, proposals, gt_instances)

        KL_fake_output = torch.nn.functional.log_softmax(fake_logits, dim=1)
        uniform_dist = torch.Tensor(KL_fake_output.size(0), self.roi_heads.num_classes+1).fill_((1./(self.roi_heads.num_classes+1))).cuda()
        # breakpoint()
        errG_KL = torch.nn.functional.kl_div(KL_fake_output, uniform_dist) * (self.roi_heads.num_classes+1)
        generator_loss = errG + 1 * errG_KL
        generator_loss.backward()
        self.optimizerG.step()

        noise = torch.FloatTensor(data.size(0), self.nz, 1, 1).normal_(0, 1).cuda()
        noise = Variable(noise)
        fake = self.G(noise).detach()
        fake_upsampled_copied = copy.deepcopy(batched_inputs)

        for index in range(len(fake)):
            fake_upsampled_copied[index]['image'] = torch.nn.functional.interpolate(
                fake[index].unsqueeze(0), list(batched_inputs[index]['image'].size())[1:], mode='bilinear') * 255
            fake_upsampled_copied[index]['image'] = fake_upsampled_copied[index]['image'].squeeze()

        fake_images = self.preprocess_image(fake_upsampled_copied)
        features_fake = self.backbone(fake_images.tensor)
        _, fake_logits = self.roi_heads.forward_special(fake_images, features_fake, proposals, gt_instances)


        KL_fake_output = torch.nn.functional.log_softmax(fake_logits, dim=1)
        uniform_dist = torch.Tensor(KL_fake_output.size(0), self.roi_heads.num_classes + 1).fill_(
            (1. / (self.roi_heads.num_classes + 1))).cuda()
        KL_loss_fake = torch.nn.functional.kl_div(KL_fake_output, uniform_dist) * (self.roi_heads.num_classes+1)


        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}

        for key in list(detector_losses.keys()):
            if key == 'loss_cls':
                detector_losses[key] += 0.05 * KL_loss_fake
                # print(KL_loss_fake)
        # for key in list(proposal_losses.keys()):
        #     proposal_losses[key] = 0.00001 * proposal_losses[key]
        losses.update(detector_losses)
        losses.update(proposal_losses)
        # losses.update({'dummy': torch.zeros(1).cuda()})
        # losses.update({'dummy': torch.zeros(1).cuda()})
        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
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
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

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

        # print(box_cls[:, -1].shape)
        # specifically used for this baseline.
        box_cls_cache = box_cls
        box_cls = torch.cat([box_cls[:, :-3], box_cls[:, -1].view(-1,1)],1)
        inter_feat = box_cls_cache
        box_cls = torch.nn.functional.softmax(box_cls, dim=-1)

        # Remove background category
        scores = box_cls[:, :-1]
        # scores = box_cls[:, :-1]

        num_bbox_reg_classes = box_delta.shape[1] // 4
        box_delta = box_delta.reshape(-1, 4)
        box_delta = box_delta.view(-1, num_bbox_reg_classes, 4)
        filter_mask = scores > self.roi_heads.box_predictor.test_score_thresh#self.test_score_thres

        filter_inds = filter_mask.nonzero(as_tuple=False)
        # breakpoint()
        if num_bbox_reg_classes == 1:
            box_delta = box_delta[filter_inds[:, 0], 0]
        else:
            # specifically for this baseline.
            box_delta = torch.cat([box_delta[:,:-3,:], box_delta[:, -1, :].unsqueeze(1)], 1)
            # breakpoint()
            box_delta = box_delta[filter_mask]

        det_labels = torch.arange(scores.shape[1], dtype=torch.long)
        det_labels = det_labels.view(1, -1).expand_as(scores)

        scores = scores[filter_mask]
        det_labels = det_labels[filter_mask]

        inter_feat = inter_feat[filter_inds[:, 0]]
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

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results