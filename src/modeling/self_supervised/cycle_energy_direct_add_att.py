import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from detectron2.structures import ImageList

from .build import SSHEAD_REGISTRY
from .ss_layers import Flatten


class CycleEnergyDirectAddAttHead(nn.Module):
    def __init__(self, cfg, cin):
        super(CycleEnergyDirectAddAttHead, self).__init__()

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
        # self.enc2 = nn.Linear(256*7*7, 256)
        # self.attention_matrix = nn.Linear(cfg.MODEL.SS.SELECTED_FRAMES*
        #                                   cfg.DATALOADER.PAIR_OFFSET_RANGE,
        #                                   cfg.MODEL.SS.SELECTED_FRAMES, bias=False)
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
        # breakpoint()
        score = F.softmax(dist/ 16, dim=1)
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
                # print(idxs)
                feat_u = self.enc1(u)
                # print(feat_u.size(0))
                feat_u = feat_u.view(feat_u.size(0), feat_u.size(1))
                if feat_u.size(0) == 0:
                    # # print(feat_u.size(), feat_v.size())
                    # loss = 0 #feat_u.sum()
                    # correct = 0
                    # cnt = 0
                    pass
                else:
                    pos_fea_cur = self.add(u).view(-1, 256 * 49)
                    if pos_fea is None:
                        pos_fea = pos_fea_cur
                    else:
                        pos_fea = torch.cat([pos_fea, pos_fea_cur], 0)
                    for frame in range(self.cfg.DATALOADER.PAIR_OFFSET_RANGE):
                        v = features[idxs[i+frame]: idxs[i + frame + 1]]
                        feat_v = self.enc1(v)
                        # print(feat_v.size(0))
                        feat_v = feat_v.view(feat_v.size(0), feat_v.size(1))
                        loss, correct, cnt, soft_target_score = self.computer_corr_softmax(feat_u, feat_v)
                        # print(soft_target_score)
                        if frame == 0:
                            neg_fea_temp = torch.matmul(soft_target_score, self.add(v).view(-1, 256 * 49))
                        else:
                            neg_fea_temp = torch.cat([neg_fea_temp,
                                                        torch.matmul(soft_target_score,
                                                                     self.add(v).view(-1, 256 * 49))], 0)
                    # if neg_fea_temp.size(0) != \
                    #         self.cfg.MODEL.SS.SELECTED_FRAMES*self.cfg.DATALOADER.PAIR_OFFSET_RANGE:
                    #     print(feat_u.shape)
                    #     print(neg_fea_temp.shape)
                    #     break
                    if neg_fea is None:
                        # breakpoint()
                        # neg_fea_trans = F.relu(self.enc2(neg_fea_temp).view(
                        #                   self.cfg.DATALOADER.PAIR_OFFSET_RANGE,
                        #                   pos_fea_cur.size(0),
                        #                    -1).permute(1,0,2))
                        # # breakpoint()
                        # pos_fea_trans = F.relu(self.enc2(pos_fea_cur).view(pos_fea_cur.size(0),1, 256))
                        # temp = torch.bmm(neg_fea_trans, pos_fea_trans)
                        #===========================
                        # temp = torch.bmm(neg_fea_temp.view(
                        #                   self.cfg.DATALOADER.PAIR_OFFSET_RANGE,
                        #                   pos_fea_cur.size(0),
                        #                    -1).permute(1,0,2), pos_fea_cur.view(pos_fea_cur.size(0),256*49, 1))
                        # ===========================
                        # temp = torch.cdist(neg_fea_temp.view(
                        #                   self.cfg.DATALOADER.PAIR_OFFSET_RANGE,
                        #                   pos_fea_cur.size(0),
                        #                    -1).permute(1,0,2), pos_fea_cur.view(pos_fea_cur.size(0),1, 256*49), p=2).pow(2)
                        # breakpoint()
                        # self.computer_corr_softmax()



                        # data1 = neg_fea_temp.view(
                        #     self.cfg.DATALOADER.PAIR_OFFSET_RANGE,pos_fea_cur.size(0),-1).permute(1, 0, 2)
                        # data2 = pos_fea_cur.view(pos_fea_cur.size(0), 1, 256 * 49).repeat(
                        #     1, self.cfg.DATALOADER.PAIR_OFFSET_RANGE,1)
                        # temp = (data1-data2).pow(2).sum(2) / (256*49)

                        # data1 = neg_fea_temp.view(
                        #     self.cfg.DATALOADER.PAIR_OFFSET_RANGE, pos_fea_cur.size(0), -1).permute(1, 0, 2)
                        # data2 = pos_fea_cur.view(pos_fea_cur.size(0), 1, 256 * 49).repeat(
                        #     1, self.cfg.DATALOADER.PAIR_OFFSET_RANGE, 1)
                        # temp = (data1 - data2).pow(2).sum(2) / (16 * 7)


                        data1 = neg_fea_temp.view(
                            self.cfg.DATALOADER.PAIR_OFFSET_RANGE, pos_fea_cur.size(0), -1).permute(1, 0, 2)
                        temp = -torch.bmm(data1, pos_fea_cur.view(pos_fea_cur.size(0), 256 * 49, 1))
                        temp = temp / (256 * 49)

                        # data1 = neg_fea_temp.view(
                        #     self.cfg.DATALOADER.PAIR_OFFSET_RANGE, pos_fea_cur.size(0), -1).permute(1, 0, 2)
                        # temp = -torch.bmm(data1, pos_fea_cur.view(pos_fea_cur.size(0), 256 * 49, 1))
                        # temp = temp / (16 * 7)

                        if temp.size(0) == 1:
                            attention_matrix = F.softmax(
                                temp.view(1, temp.size(1)),1)
                        else:
                            attention_matrix = F.softmax(
                                temp.squeeze(), 1)#.detach()
                        # breakpoint()
                        # print(attention_matrix)
                        neg_fea = attention_matrix.unsqueeze(2) * neg_fea_temp.view(
                            self.cfg.DATALOADER.PAIR_OFFSET_RANGE,
                            pos_fea_cur.size(0),-1).permute(1,0,2)
                        neg_fea = neg_fea.sum(1).squeeze()
                        if len(neg_fea) == 12544:
                            neg_fea = neg_fea.unsqueeze(0)
                        # breakpoint()
                    else:
                        neg_fea = torch.cat(
                            [neg_fea, torch.matmul(F.softmax(
                                self.attention_matrix.weight, 1), neg_fea_temp)], 0)
                        # breakpoint()
                # assert frame ==  - 1
                prev = idxs[i + self.cfg.DATALOADER.PAIR_OFFSET_RANGE]


        if pos_fea is not None:
            assert len(pos_fea) == len(neg_fea), len(neg_fea)
            # print('total loss: {:.4f}\ttotal acc: {:.3f}'.format(total_loss, total_acc))
            return {'loss_cycle': 0.0}, 0.0, torch.cat([pos_fea, neg_fea], 0), None
        else:
            print('marker!')
            conv_input = torch.zeros(256,256,5,5).cuda()
            fake_loss = (self.enc1(conv_input) - self.enc1(conv_input)).sum() + \
                        (self.add(conv_input) - self.add(conv_input)).sum()
                        # (self.enc2(torch.zeros(256*7*7).cuda())-self.enc2(torch.zeros(256*7*7).cuda())).sum()
            assert fake_loss == 0
            # print('fake_loss: ', fake_loss)
            return {'loss_cycle': 0.0}, 0.0, None, fake_loss


@SSHEAD_REGISTRY.register()
def build_cycle_energy_direct_add_att_head(cfg, input_shape):
    in_channels = cfg.MODEL.FPN.OUT_CHANNELS
    rot_head = CycleEnergyDirectAddAttHead(cfg, in_channels)
    return rot_head
