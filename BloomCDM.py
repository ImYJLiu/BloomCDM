import torch
import torch.nn as nn
import torch.nn.functional as F


class BloomCDM(nn.Module):  # 定义Bloom模型,将R矩阵传入bloomCDM中~~
    def __init__(self, user_size, item_size, item_size_1, item_size_2, batch_size, know_group, higher_know_group, dim,
                 weight_decay, weight_decay_1, lamdaU, lamdaV_1):
        super().__init__()
        self.W = nn.Parameter(torch.empty(user_size, dim))
        self.H = nn.Parameter(torch.empty(item_size, dim))
        self.H_1 = nn.Parameter(torch.empty(item_size_1, dim))  # 最大的知识点的个数item_size_1
        self.H_2 = nn.Parameter(torch.empty(item_size_2, dim))  # 最大的知识点的个数item_size_1
        self.R_1 = know_group
        self.R_2 = higher_know_group
        nn.init.xavier_normal_(self.W.data)
        nn.init.xavier_normal_(self.H.data)
        nn.init.xavier_normal_(self.H_1.data)
        nn.init.xavier_normal_(self.H_2.data)
        self.dim = dim
        self.batch_size = batch_size
        self.item_size = item_size
        self.item_size_1 = item_size_1
        self.user_size = user_size
        self.weight_decay = weight_decay
        self.weight_decay_1 = weight_decay_1
        self.lamdaU = lamdaU
        self.lamdaV_1 = lamdaV_1

    def getSumR_1(self, u, i_1):
        r_1 = torch.zeros(len(u))
        for i in range(len(u)):
            r_1[i] = self.R_1[u[i].item()][i_1[i].item()]['avg_sum']
        return r_1

    def getSumR_2(self, u, i_2):
        r_2 = torch.zeros(len(u))
        for i in range(len(u)):
            r_2[i] = self.R_2[u[i].item()][i_2[i].item()]['avg_sum']
        return r_2

    # 第一层 u, i j
    # 第二层 u, i_1
    def forward(self, u, i, j, i_1, i_2, batch_size):
        """Return loss value.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]

        Returns:
            torch.FloatTensor
        """
        r_1 = self.getSumR_1(u, i_1)
        r_2 = self.getSumR_2(u, i_2)
        i_1 = self.H_1[i_1]
        i_2 = self.H_2[i_2]
        u = self.W[u, :]
        i = self.H[i, :]
        j = self.H[j, :]
        x_ui = torch.mul(u, i).sum(dim=1)
        x_ui_1 = torch.mul(u, i_1).sum(dim=1)
        x_ui_2 = torch.mul(u, i_2).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).sum() - ((r_1.cuda() - x_ui_1) ** 2).sum() - (
                    (r_2.cuda() - x_ui_2) ** 2).sum()
        regularization = self.lamdaU * (u.norm(dim=1) ** 2).sum() + \
                         self.lamdaV_1 * (i_1 - i).norm(dim=1).pow(2).sum() + \
                         self.lamdaV_1 * (i_2 - i_1).norm(dim=1).pow(2).sum()

        # #BPR
        # log_prob = F.logsigmoid(x_uij).sum()
        # regularization = (self.lamdaU * (u.norm(dim=1) ** 2).sum() + self.lamdaU * (i.norm(dim=1) ** 2).sum() + self.lamdaU * (j.norm(dim=1) ** 2).sum()) / batch_size

        # #BloomCDM_RC
        # log_prob = F.logsigmoid(x_uij).sum() - ((r_1.cuda() - x_ui_1) ** 2).sum()
        # regularization = self.lamdaU * (u.norm(dim=1) ** 2).sum() + \
        #                  self.lamdaV_1 * (i_1 - i).norm(dim=1).pow(2).sum()


        return (-log_prob + regularization)/batch_size


    def recommend(self, u):
        """Return recommended item list given users.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]

        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """
        u = self.W[u, :]
        x_ui = torch.mm(u, self.H.t())
        pred = torch.argsort(x_ui, dim=1)
        return pred