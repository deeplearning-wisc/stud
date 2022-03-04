import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.structures import ImageList

from .build import SSHEAD_REGISTRY
from .ss_layers import Flatten


class CycleEnergyDirectAddNoHead(nn.Module):
    def __init__(self, cfg, cin):
        super(CycleEnergyDirectAddNoHead, self).__init__()

        self.name = 'cycle'
        self.input = 'ROI'
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.coef = cfg.MODEL.SS.COEF
        
        # self.enc1 = nn.Sequential(
        #     nn.Conv2d(cin, 256, kernel_size=3, padding=0, bias=True),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=0, bias=True),
        #     # nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d(1)
        #     # nn.Flatten(start_dim=1, end_dim=-1)
        # )
        # self.add = nn.Conv2d(256, 256, kernel_size=1)
        # self.map_back = nn.Linear(256, 256*49)

        self.topk = 100
        self.bs = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.scale = cfg.MODEL.SS.LOSS_SCALE

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

        uu = feat_u.unsqueeze(1).repeat(1, vs, 1).view(-1, fs)
        vv = feat_v.repeat(us, 1)

        diff = uu - vv
        dist = (diff * diff).sum(dim=1).view(us, vs) * self.coef
        score = F.softmax(dist, dim=1)
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
        total_loss = 0.0
        corrects = 0
        counts = 0
        pos_fea= None
        neg_fea = None
        prev = 0
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
            for i in range(0, len(idxs), 2):
                u = features[prev:idxs[i]]
                v = features[idxs[i]: idxs[i+1]]
                prev = idxs[i+1]
                feat_u = u.view(-1, 256*49)#self.enc1(u)
                feat_v = v.view(-1, 256*49)#self.enc1(v)
                feat_u = feat_u.view(feat_u.size(0), feat_u.size(1))
                feat_v = feat_v.view(feat_v.size(0), feat_v.size(1))
                if feat_u.size(0) == 0:
                    print(feat_u.size(), feat_v.size())
                    loss = feat_u.sum()
                    correct = 0
                    cnt = 0
                else:
                    loss, correct, cnt, soft_target_score = self.computer_corr_softmax(feat_u, feat_v)
                    # breakpoint()
                    if pos_fea is None:
                        pos_fea = u.view(-1, 256*49)
                        neg_fea = torch.matmul(soft_target_score, v.view(-1, 256*49))
                        # breakpoint()
                    else:
                        pos_fea = torch.cat([pos_fea, u.view(-1, 256*49)], 0)
                        neg_fea = torch.cat([neg_fea, torch.matmul(soft_target_score, v.view(-1, 256*49))], 0)

                total_loss += loss*cnt
                corrects += correct
                counts += cnt
            # breakpoint()
            if counts != 0:
                total_loss /= counts
                total_acc = corrects/counts
            else:
                total_acc = 0.
        if pos_fea is not None:
            assert len(pos_fea) == len(neg_fea)
            # print('total loss: {:.4f}\ttotal acc: {:.3f}'.format(total_loss, total_acc))
            return {'loss_cycle': total_loss * self.scale}, total_acc, torch.cat([pos_fea, neg_fea], 0)
        else:
            return {'loss_cycle': total_loss * self.scale}, total_acc, None


@SSHEAD_REGISTRY.register()
def build_cycle_energy_direct_no_head(cfg, input_shape):
    in_channels = cfg.MODEL.FPN.OUT_CHANNELS
    rot_head = CycleEnergyDirectAddNoHead(cfg, in_channels)
    return rot_head
