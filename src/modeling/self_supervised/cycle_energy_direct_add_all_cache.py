import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.structures import ImageList

from .build import SSHEAD_REGISTRY
from .ss_layers import Flatten


class CycleEnergyDirectAddAllHead(nn.Module):
    def __init__(self, cfg, cin):
        super(CycleEnergyDirectAddAllHead, self).__init__()

        self.name = 'cycle'
        self.input = 'ROI'
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.coef = cfg.MODEL.SS.COEF

        self.enc1 = nn.Sequential(
            nn.Conv2d(cin, 256, kernel_size=3, padding=0, bias=True),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=0, bias=True),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
            # nn.Flatten(start_dim=1, end_dim=-1)
        )
        self.add = nn.Conv2d(256, 256, kernel_size=1)
        # self.map_back = nn.Linear(256, 256*49)

        self.topk = 100
        self.bs = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.scale = cfg.MODEL.SS.LOSS_SCALE
        self.cfg = cfg

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)

    def cal_pair_dist(self, feat_u, feat_v):
        # finding the similarity score of feat_v
        us = feat_u.size(0)
        vs = feat_v.size(0)
        fs = feat_u.size(1)
        assert fs == feat_v.size(1)

        dist = torch.cdist(feat_u, feat_v, p=2).pow(2) * self.coef
        # breakpoint()
        # uu = feat_u.unsqueeze(1).repeat(1, vs, 1).view(-1, fs)
        # vv = feat_v.repeat(us, 1)
        #
        # diff = uu - vv
        # dist = (diff * diff).sum(dim=1).view(us, vs) * self.coef
        score = F.softmax(dist / 16, dim=1)
        return dist, score

    def computer_corr_softmax(self, feat_u, feat_v):
        # track forward
        # calculate the L2 distance between feat_u and feat_v

        sim_dist, sim_score = self.cal_pair_dist(feat_u, feat_v)
        # soft_v = torch.matmul(sim_score, feat_v)
        #
        # # track backward
        # back_dist, back_score = self.cal_pair_dist(soft_v, feat_u)
        # labels = torch.arange(len(feat_u)).long().to(back_dist.device)
        # loss = nn.CrossEntropyLoss()(back_dist, labels)
        #
        # if back_dist.size(1) == 0:# there is no objects in the first frame.
        #     print(back_dist.size(), feat_u.size(), feat_v.size(), loss)
        # correct = (back_dist.argmax(dim=1) == labels).float().sum()
        # count = len(back_dist)
        return torch.zeros(1).cuda(), 0, 0, sim_score


    def forward(self, features, prev_boxes=None):
        features, idxs, proposals = features
        pos_fea= None
        neg_fea = None
        prev = 0
        # frame = None
        # since the number of proposals might be different for different pairs
        if prev_boxes is not None:
            feat_u = self.enc1(features)
            feat_v = self.enc1(prev_boxes)
            feat_u = feat_u.view(feat_u.size(0), feat_u.size(1))
            feat_v = feat_v.view(feat_v.size(0), feat_v.size(1))
            if feat_u.size(0) == 0:
                print(feat_u, feat_v)
                return {'loss_cycle': feat_u.sum() * self.scale}, 0.
            total_loss, correct, cnt, _ = self.computer_corr_softmax(feat_u, feat_v)
            # print('correct: ', correct, 'cnt: ', cnt)
            total_acc = correct.item()/cnt

        else:
            for i in range(0, len(idxs), self.cfg.DATALOADER.PAIR_OFFSET_RANGE + 1):
                u = features[prev:idxs[i]]
                feat_u = self.enc1(u)
                feat_u = feat_u.view(feat_u.size(0), feat_u.size(1))
                if feat_u.size(0) == 0:
                    # # print(feat_u.size(), feat_v.size())
                    # loss = 0 #feat_u.sum()
                    # correct = 0
                    # cnt = 0
                    pass
                else:
                    if pos_fea is None:
                        pos_fea = self.add(u).view(-1, 256 * 49)
                    else:
                        pos_fea = torch.cat([pos_fea, self.add(u).view(-1, 256 * 49)], 0)
                    for frame in range(self.cfg.DATALOADER.PAIR_OFFSET_RANGE):
                        v = features[idxs[i+frame]: idxs[i + frame + 1]]
                        feat_v = self.enc1(v)
                        feat_v = feat_v.view(feat_v.size(0), feat_v.size(1))
                        loss, correct, cnt, soft_target_score = self.computer_corr_softmax(feat_u, feat_v)
                        if neg_fea is None:
                            neg_fea = torch.matmul(soft_target_score, self.add(v).view(-1, 256 * 49))
                            # breakpoint()
                        else:
                            neg_fea = torch.cat(
                                [neg_fea, torch.matmul(soft_target_score, self.add(v).view(-1, 256 * 49))], 0)
                # assert frame ==  - 1
                prev = idxs[i + self.cfg.DATALOADER.PAIR_OFFSET_RANGE]
        # breakpoint()
        if pos_fea is not None:
            assert len(pos_fea) == len(neg_fea) / self.cfg.DATALOADER.PAIR_OFFSET_RANGE
            # print('total loss: {:.4f}\ttotal acc: {:.3f}'.format(total_loss, total_acc))
            return {'loss_cycle': 0.0}, 0.0, torch.cat([pos_fea, neg_fea], 0), None
        else:
            print('marker!')
            conv_input = torch.zeros(256,256, 5, 5).cuda()
            fake_loss = (self.enc1(conv_input) - self.enc1(conv_input)).sum() + \
                        (self.add(conv_input) - self.add(conv_input)).sum()
            assert fake_loss == 0
            # print('fake_loss: ', fake_loss)
            return {'loss_cycle': 0.0}, 0.0, None, fake_loss


@SSHEAD_REGISTRY.register()
def build_cycle_energy_direct_add_all_head(cfg, input_shape):
    in_channels = cfg.MODEL.FPN.OUT_CHANNELS
    rot_head = CycleEnergyDirectAddAllHead(cfg, in_channels)
    return rot_head
