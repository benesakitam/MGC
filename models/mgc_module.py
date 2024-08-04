# encoding: utf-8
import torch
import math
from torch.nn import Module
import torch.nn.functional as F
from . import resnet_mgc as resnet
import torch.nn as nn


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, temp, reduction="batchmean"):
        super().__init__()
        self.temp = temp
        self.reduction = reduction

    def forward(self, f_s, f_t):
        p_s = F.log_softmax(f_s / self.temp, dim=1)
        p_t = F.softmax(f_t / self.temp, dim=1)
        loss = F.kl_div(p_s, p_t, reduction=self.reduction) * (self.temp**2)
        return loss


class MGC(Module):
    def __init__(self, param_momentum, total_iters):
        super(MGC, self).__init__()
        self.total_iters = total_iters
        self.param_momentum = param_momentum
        self.current_train_iter = 0

        self.student_encoder = resnet.resnet50(
            low_dim=256, width=1, hidden_dim=4096, MLP="cyclemlp", CLS=False, bn="torchsync", predictor=True
        )
        self.teacher_encoder = resnet.resnet50(
            low_dim=256, width=1, hidden_dim=4096, MLP="cyclemlp", CLS=False, bn="torchsync", predictor=False
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.T = 0.2
        # todo
        self.lmd_cc = 1.0
        self.lmd_kl_pa = 200
        self.lmd_kl_aa = 50
        # self.fea_loss = torch.nn.MSELoss()
        self.kl_loss = DistillKL(temp=4.0)

        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        self.momentum_update(m=0)

    @torch.no_grad()
    def momentum_update(self, m):
        for p1, p2 in zip(self.student_encoder.named_parameters(), self.teacher_encoder.named_parameters()):
            flag = 'fc' in p1[0]
            if not flag:
                # p2.data.mul_(m).add_(1 - m, p1.detach().data)
                p2[1].data = m * p2[1].data + (1.0 - m) * p1[1].detach().data

        for p1, p2 in zip(self.student_encoder.fc1.parameters(),
                          self.teacher_encoder.fc1.parameters()):
            p2.data = m * p2.data + (1.0 - m) * p1.detach().data
        for p1, p2 in zip(self.student_encoder.fc2.parameters(),
                          self.teacher_encoder.fc2.parameters()):
            p2.data = m * p2.data + (1.0 - m) * p1.detach().data

    def get_param_momentum(self):
        return 1.0 - (1.0 - self.param_momentum) * (
            (math.cos(math.pi * self.current_train_iter / self.total_iters) + 1) * 0.5
        )

    def forward(self, inps, update_param=True):
        if update_param:
            current_param_momentum = self.get_param_momentum()
            self.momentum_update(current_param_momentum)

        x1, x2 = inps[0], inps[1]
        q1, att_q1 = self.student_encoder(x1)
        q2, att_q2 = self.student_encoder(x2)

        with torch.no_grad():
            k1, att_k1 = self.teacher_encoder(x1)
            k2, att_k2 = self.teacher_encoder(x2)

        # same branch contrastive learning
        con_c2c_loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        con_m2m_loss = self.contrastive_loss(att_q1, att_k2) + self.contrastive_loss(att_q2, att_k1)

        # cross branch contrastive learning
        con_c2m_loss = self.contrastive_loss(q1, att_k2) + self.contrastive_loss(q2, att_k1)
        con_m2c_loss = self.contrastive_loss(att_q1, k2) + self.contrastive_loss(att_q2, k1)

        # positive pairs advancement: teacher -> student
        kl_loss_t2s = self.kl_loss(q1, k2.detach()) + self.kl_loss(att_q1, att_k2.detach()) \
                      + self.kl_loss(q2, k1.detach()) + self.kl_loss(att_q2, att_k1.detach())

        # attention domain advancement: mlp -> cnn
        kl_loss_m2c = self.kl_loss(q1, att_q1.detach()) + self.kl_loss(q2, att_q2.detach()) \
                      + self.kl_loss(k1, att_k1.detach()) + self.kl_loss(k2, att_k2.detach())

        loss = con_c2c_loss + con_m2m_loss\
               + self.lmd_cc * (con_c2m_loss + con_m2c_loss) \
               + self.lmd_kl_pa * kl_loss_t2s \
               + self.lmd_kl_aa * kl_loss_m2c

        self.current_train_iter += 1
        if self.training:
            return loss, con_c2c_loss, con_m2m_loss, \
                   self.lmd_cc * con_c2m_loss, self.lmd_cc * con_m2c_loss, \
                   self.lmd_kl_pa * kl_loss_t2s, \
                   self.lmd_kl_aa * kl_loss_m2c

    def contrastive_loss(self, q, k):
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
