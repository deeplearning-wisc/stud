import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import SSHEAD_REGISTRY
from .ss_layers import Bottleneck, conv1x1, conv3x3
from ..utils.image_list import ImageList, crop_tensor


class RotationHead(nn.Module):
    def __init__(self, cfg, cin):
        super(RotationHead, self).__init__()

        # resnet config
        self.name = 'rot'
        self.input = 'images'
        self.device = torch.device(cfg.MODEL.DEVICE)
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # hard code the task specific parameters in order to
        # support multi-tasking
        self.crop_size = 224
        # self.ratio = 2
        self.ratio = cfg.MODEL.SS.RATIO  # crop image ratio

        depth = cfg.MODEL.RESNETS.DEPTH
        stage_ids = {"res2": 0, "res3": 1, "res4": 2, "res5": 3}
        num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3],
                                152: [3, 8, 36, 3]}[depth]
        self.start_stage = min(stage_ids[cfg.MODEL.SS.FEAT_LEVEL]+1, 3)
        self.inplanes = cin
        self.scale = cfg.MODEL.SS.LOSS_SCALE

        out_channels = self.inplanes

        for i in range(self.start_stage, 4):
            out_channels *= 2
            setattr(self, "layer{}".format(i),
                    self._make_layer(Bottleneck, out_channels//4,
                                     num_blocks_per_stage[i], stride=2))

        # num_classes = cfg.MODEL.SS.NUM_CLASSES
        num_classes = 4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, batched_inputs, feat_base, feat_level):
        x, y = self.gen_ss_inputs(batched_inputs)
        x = feat_base(x)[feat_level]
        for i in range(self.start_stage, 4):
            x = getattr(self, "layer{}".format(i))(x)

        x = self.avgpool(x)
        bs = x.size(0)
        x = x.squeeze()
        if bs == 1:
            x = x.unsqueeze(0)
        x = self.fc(x)
        loss = self.criterion(x, y.long())
        losses = {'loss_rot_cls': loss * self.scale}
        return x, y, losses

    # add the data processing for each task
    def preprocess_image_ss(self, batched_inputs):
        """resize and random crop the images"""
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors_crop(images, self.crop_size, self.ratio)
        return images

    def gen_ss_inputs(self, batched_inputs):
        """produce rotation targets"""
        images = self.preprocess_image_ss(batched_inputs=batched_inputs)
        tensors = images.tensor.clone().to(self.device)
        targets = torch.zeros(len(tensors)).long().to(self.device)
        for i in range(len(tensors)):
            tar = np.random.choice(4)
            targets[i] = tar
            t = images.tensor[i]
            rot = t.rot90(tar, (1, 2))
            tensors[i] = rot
        images.tensor = tensors
        return tensors, targets


@SSHEAD_REGISTRY.register()
def build_rotation_head(cfg, input_shape):
    in_channels = input_shape[cfg.MODEL.SS.FEAT_LEVEL].channels
    rot_head = RotationHead(cfg, in_channels)
    return rot_head