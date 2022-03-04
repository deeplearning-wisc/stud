import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.structures import ImageList

from .build import SSHEAD_REGISTRY
from .ss_layers import Flatten


class CycleEnergyDirectAddAllNoiseHead(nn.Module):
    def __init__(self, cfg, cin):
        super(CycleEnergyDirectAddAllNoiseHead, self).__init__()

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
        self.add = nn.Conv2d(256, 256, kernel_size=1)
        # self.map_back = nn.Linear(256, 256*49)
        self.pos = []
        self.neg = []


        self.topk = 100
        self.bs = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.scale = cfg.MODEL.SS.LOSS_SCALE
        self.cfg = cfg
        self.save = []

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
        if 'vis' in self.cfg.DATASETS.TRAIN[0]:
            score = F.softmax(dist, dim=1)
        else:
            score = F.softmax(dist / 16, dim=1)
        # print(score)
        # breakpoint()
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
        return torch.zeros(1).cuda(), 0, 0, sim_score, sim_dist


    def forward(self, roi_head, features, prev_boxes=None):
        features, idxs, proposals = features
        pos_fea= None
        neg_fea = None
        fea_v_all = None
        v_all = None
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
            for i in range(0, len(idxs), self.cfg.DATALOADER.SELCTED_NUMBER + 1):
                u = features[prev:idxs[i]]
                # feat_u = self.enc1(u)
                # feat_u = feat_u.view(feat_u.size(0), feat_u.size(1))
                if u.size(0) == 0:
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
                    for frame in range(self.cfg.DATALOADER.SELCTED_NUMBER):
                        v = features[idxs[i+frame]: idxs[i + frame + 1]]
                        # feat_v = self.enc1(v)
                        # feat_v = feat_v.view(feat_v.size(0), feat_v.size(1))
                        # fea_temp = roi_head.box_head(v)
                        # predictions1 = roi_head.box_predictor(fea_temp)
                        # energy_scores_all = torch.logsumexp(predictions1[0][:, :-1], dim=1)
                        # selected_indices = energy_scores_all.argsort()[
                        #                    int(self.cfg.MODEL.SS.FILTERING1 * len(energy_scores_all)):
                        #                    int(self.cfg.MODEL.SS.FILTERING2 * len(energy_scores_all))]
                        # breakpoint()
                        # feat_v = feat_v[selected_indices]
                        # v = v[selected_indices]

                        if fea_v_all is not None:
                            # fea_v_all = torch.cat([fea_v_all, feat_v], 0)
                            v_all = torch.cat([v_all, v], 0)
                        else:
                            # fea_v_all = feat_v
                            v_all = v
                    # breakpoint()
                    # loss, correct, cnt, soft_target_score, dist = self.computer_corr_softmax(feat_u, fea_v_all)
                    # # temp
                    # fea_temp = roi_head.box_head(torch.matmul(soft_target_score, self.add(v_all).view(-1, 256 * 49)).view(-1, 256, 7, 7))
                    # predictions1 = roi_head.box_predictor(fea_temp)
                    # energy_scores_all = torch.logsumexp(predictions1[0][:, :-1], dim=1)
                    # # print(energy_scores_all)
                    #
                    # fea_temp = roi_head.box_head(u)
                    # predictions1 = roi_head.box_predictor(fea_temp)
                    # energy_scores_all_pos = torch.logsumexp(predictions1[0][:, :-1], dim=1)
                    # # print(energy_scores_all_pos)
                    #
                    # if len(self.pos) < 200:
                    #     self.pos.append(energy_scores_all_pos)
                    #     self.neg.append(energy_scores_all)
                    # else:
                    #     np.save('./pos_energy.npy', self.pos)
                    #     np.save('./neg_energy.npy', self.neg)
                    #     break
                    # # temp
                    # # breakpoint()

                    # print(energy_scores_all)
                    # print(energy_scores_all_pos)
                    # breakpoint()


                    if neg_fea is None:
                        scale = pos_fea.detach()
                        # breakpoint()
                        neg_fea = torch.tensor(0).expand(pos_fea.size()).cuda().float().normal_() * scale
                        #torch.matmul(soft_target_score, self.add(v_all).view(-1, 256 * 49))
                    else:
                        scale = pos_fea.detach()
                        neg_fea1 = torch.tensor(0).expand(pos_fea.size()).cuda().float().normal_() * scale
                        neg_fea = torch.cat(
                            [neg_fea, neg_fea1], 0)
                    # breakpoint()
                # assert frame ==  - 1
                prev = idxs[i + self.cfg.DATALOADER.SELCTED_NUMBER]
        # breakpoint()
        if pos_fea is not None:
            # print('hhh')
            assert len(pos_fea) == len(neg_fea) #/ self.cfg.DATALOADER.SELCTED_NUMBER
            # print('total loss: {:.4f}\ttotal acc: {:.3f}'.format(total_loss, total_acc))
            return {'loss_cycle': 0.0}, 0.0, torch.cat([pos_fea, neg_fea], 0), None
        else:
            print('marker!')
            conv_input = torch.zeros(256,256, 5, 5).cuda()
            fake_loss = (self.add(conv_input) - self.add(conv_input)).sum()
            assert fake_loss == 0
            # print('fake_loss: ', fake_loss)
            return {'loss_cycle': 0.0}, 0.0, None, fake_loss

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


@SSHEAD_REGISTRY.register()
def build_cycle_energy_direct_add_all_noise_head(cfg, input_shape):
    in_channels = cfg.MODEL.FPN.OUT_CHANNELS
    rot_head = CycleEnergyDirectAddAllNoiseHead(cfg, in_channels)
    return rot_head
