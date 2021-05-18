import torch
import torch.nn as nn
import torch.nn.functional as F


class PMF(nn.Module):
    def __init__(self, item_size, user_size, batch_size, dim, lamdaU, lamdaV_1):
        super().__init__()
        self.W = nn.Parameter(torch.empty(user_size, dim))
        self.H = nn.Parameter(torch.empty(item_size, dim))
        nn.init.xavier_normal_(self.W.data)
        nn.init.xavier_normal_(self.H.data)
        self.dim = dim
        self.batch_size = batch_size
        self.item_size = item_size
        self.user_size = user_size
        self.lamdaU = lamdaU
        self.lamdaV_1 = lamdaV_1

    def forward(self, u, i, s):
        """Return loss value.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]

        Returns:
            torch.FloatTensor
        """
        u = self.W[u, :]
        i = self.H[i, :]
        mult_result=torch.mm(u, i.t())
        x_ui = torch.mul(torch.eye(mult_result.shape[0]).cuda(), mult_result).sum(dim = 1)
        log_prob = ((s.float().cuda() - F.logsigmoid(x_ui)) ** 2).sum()
        regularization = self.lamdaU * (u.norm(dim=1) ** 2).sum() + self.lamdaV_1 * (i.norm(dim=1) ** 2).sum()

        return (log_prob + regularization) / self.batch_size

