import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoreFusion(nn.Module):
    def __init__(self, opt):
        super(CoreFusion, self).__init__()
        self.opt = opt
        self.class_num = 2
        self.linear_v = nn.Linear(self.opt['fusion']['dim_v'], self.opt['fusion']['dim_hv'])
        self.linear_t = nn.Linear(self.opt['fusion']['dim_t'], self.opt['fusion']['dim_ht'])
        self.bn_layer_v = nn.BatchNorm1d(self.opt['fusion']['dim_hv'], affine=True, track_running_stats=True)
        self.bn_layer_t = nn.BatchNorm1d(self.opt['fusion']['dim_ht'], affine=True, track_running_stats=True)

        self.list_linear_hv = nn.ModuleList([
            nn.Linear(self.opt['fusion']['dim_hv'], self.opt['fusion']['dim_mm'])
            for i in range(self.opt['fusion']['R'])])

        self.list_linear_ht = nn.ModuleList([
            nn.Linear(self.opt['fusion']['dim_ht'], self.opt['fusion']['dim_mm'])
            for i in range(self.opt['fusion']['R'])])

        self.dist_learning_v = nn.Linear(self.opt['fusion']['dim_hv'], self.opt['fusion']['dim_hv'])
        self.dist_learning_t = nn.Linear(self.opt['fusion']['dim_ht'], self.opt['fusion']['dim_ht'])

        self.fake_ln1 = nn.Linear(self.opt['fusion']['dim_mm'], self.opt['fake_dec']['hidden1'])
        self.fake_last = nn.Linear(self.opt['fake_dec']['hidden1'], self.class_num)

        self.bn_layer1 = nn.BatchNorm1d(self.opt['fake_dec']['hidden1'], affine=True, track_running_stats=True)
        self.bn_layer4 = nn.BatchNorm1d(self.class_num, affine=True, track_running_stats=True)

        self.linear_calculsim = nn.Linear(self.opt['fusion']['dim_mm'], 1)

    def _calculsim(self, x):
        batch_size_v = x.size(0)
        batch_size_t = x.size(1)

        x = F.dropout(x,
                      p=self.opt['classif']['dropout'],
                      training=self.training)
        x = self.linear_calculsim(x)
        x = torch.sigmoid(x)
        x = x.view(batch_size_v, batch_size_t)
        return x

    def forward(self, input_v, input_t):
        batch_size_v = input_v.size(0)
        batch_size_t = input_t.size(0)

        input_v = torch.relu(input_v)

        x_v = self.linear_v(input_v)
        x_v = self.bn_layer_v(x_v)
        x_v = F.relu(x_v)

        x_t = self.linear_t(input_t)
        x_t = self.bn_layer_t(x_t)
        x_t = F.relu(x_t)

        x_dl_v = self.dist_learning_v(x_v)
        x_dl_t = self.dist_learning_t(x_t)

        x_mm = []
        for i in range(self.opt['fusion']['R']):

            x_hv = F.dropout(x_v, p=self.opt['fusion']['dropout_hv'], training=self.training)
            x_hv = self.list_linear_hv[i](x_hv)

            x_ht = F.dropout(x_t, p=self.opt['fusion']['dropout_ht'], training=self.training)
            x_ht = self.list_linear_ht[i](x_ht)

            x_mm.append(torch.mul(x_hv[:, None, :], x_ht[None, :, :]))

        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size_v, batch_size_t, self.opt['fusion']['dim_mm'])

        pairs_num = min(batch_size_v, batch_size_t)
        x_mm_diagonal = x_mm[torch.arange(pairs_num), torch.arange(pairs_num), :]

        fake_res = self.fake_ln1(x_mm_diagonal)
        fake_res = self.bn_layer1(fake_res)
        fake_res = F.relu(fake_res)

        fake_res = self.fake_last(fake_res)
        fake_res = self.bn_layer4(fake_res)
        fake_res = F.softmax(fake_res, dim=1)

        sim = self._calculsim(x_mm)

        return sim, x_v, x_t, x_dl_v, x_dl_t, x_mm_diagonal, fake_res


def factory(opt, cuda=True):
    opt = copy.copy(opt)
    model = CoreFusion(opt)
    if cuda:
        model = model.cuda()
    return model
