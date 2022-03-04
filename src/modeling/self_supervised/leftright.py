import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from detectron2.structures import ImageList, crop_tensor

import os

from .build import SSHEAD_REGISTRY
from .ss_layers import Bottleneck, conv1x1, conv3x3


class LeftRightHead(nn.Module):
    def __init__(self, cfg, cin):
        super(LeftRightHead, self).__init__()

        # resnet config
        self.name = 'leftright'
        self.input = 'images'
        self.device = torch.device(cfg.MODEL.DEVICE)
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.crop_size = cfg.MODEL.SS.CROP_SIZE
        self.ratio = cfg.MODEL.SS.RATIO  # crop image ratio
        self.add_norm = cfg.MODEL.SS.JIGSAW.NORM

        depth = cfg.MODEL.RESNETS.DEPTH
        stage_ids = {"res2": 0, "res3": 1, "res4": 2, "res5": 3}
        num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3],
                                152: [3, 8, 36, 3]}[depth]
        self.start_stage = min(stage_ids[cfg.MODEL.SS.FEAT_LEVEL]+1, 3)
        self.inplanes = cin
        self.scale = cfg.MODEL.SS.LOSS_SCALE

        out_channels = self.inplanes

        self.fusion = nn.Sequential(
            nn.Conv2d(cin*2, cin, kernel_size=3, stride=1,
                      padding=1,  bias=False),
            nn.ReLU(inplace=True))

        for i in range(self.start_stage, 4):
            out_channels *= 2
            setattr(self, "layer{}".format(i),
                    self._make_layer(Bottleneck, out_channels//4,
                                     num_blocks_per_stage[i], stride=2))

        num_classes = cfg.MODEL.SS.NUM_CLASSES
        assert num_classes == 2

        self.class_file = cfg.MODEL.SS.CLASS_FILE
        self.permutations = self.__retrive_permutations(num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, num_classes))
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
        B, T, C, H, W = x.size()
        x = x.transpose(0, 1)
        x_list = []

        for t in range(2):
            z = feat_base(x[t])[feat_level]  # shared feature backbone
            x_list.append(z)

        x = torch.cat(x_list, dim=1)
        # print('x size: ', x.size())
        x = self.fusion(x)

        for i in range(self.start_stage, 4):
            x = getattr(self, "layer{}".format(i))(x)
        x = self.avgpool(x)
        x = self.classifier(x.view(B, -1))

        loss = self.criterion(x, y.long())
        losses = {'loss_leftright_cls': loss * self.scale}

        # print('ss pred accuracy: ',
        #       (x.argmax(axis=1)==y).float().mean().item()*100)
        # print('softmax x: ', F.softmax(x, dim=1))
        return x, y, losses

    # add the data processing for each task
    def preprocess_image_ss(self, batched_inputs):
        """resize and random crop the images"""
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        # randomly crop an image patch of 255
        images = ImageList.from_tensors_crop(images, 500, self.ratio)
        return images

    def gen_single_input(self, input_tensor):
        # print('input_tensor size: ', input_tensor.size())
        s = int(float(input_tensor.size(1)) / 2)
        tiles = [None] * 2
        for n in range(2):
            c = [n * s, (n + 1) * s]
            tile = input_tensor[:, 0:s, c[0]:c[1]]
            # tile = input_tensor.crop(c.tolist())
            tile = crop_tensor(tile, (224, 224))
            if self.add_norm:
                # Normalize the patches indipendently to avoid low level features shortcut
                # print('tile size: ', tile.size())
                m, std = tile.reshape(3, -1).mean(dim=1).cpu().numpy(), tile.reshape(3, -1).std(dim=1).cpu().numpy()
                std[std == 0] = 1
                norm = transforms.Normalize(mean=m.tolist(), std=std.tolist())
                tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))

        data = [tiles[self.permutations[order][t]] for t in range(2)]
        data = torch.stack(data, 0)

        return data, int(order)

    def gen_ss_inputs(self, batched_inputs):
        """produce rotation targets"""
        images = self.preprocess_image_ss(batched_inputs=batched_inputs)
        tensors = images.tensor.clone().to(self.device)
        targets = torch.zeros(len(tensors)).long().to(self.device)
        tiles = []

        for i in range(len(tensors)):
            data, tar = self.gen_single_input(tensors[i])
            tiles.append(data)
            targets[i] = tar

        stacked_inputs = torch.stack(tiles, 0)

        return stacked_inputs, targets

    def __retrive_permutations(self, classes):
        all_perm = np.load(os.path.join(self.class_file,
                                        'permutations_hamming_max_{}.npy'.format(classes)))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

@SSHEAD_REGISTRY.register()
def build_leftright_head(cfg, input_shape):
    in_channels = input_shape[cfg.MODEL.SS.FEAT_LEVEL].channels
    # print('in_channels: ', in_channels)
    rot_head = LeftRightHead(cfg, in_channels)
    return rot_head
