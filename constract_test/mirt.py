import torch
import torch.nn as nn
import torch.nn.functional as F


class MIRT(nn.Module):
    def __init__(self, item_size, user_size, batch_size, k):
        super().__init__()
        self.diff = nn.Parameter(torch.empty(item_size, 1))
        self.disc = nn.Parameter(torch.empty(item_size, k))
        self.theta = nn.Parameter(torch.empty(user_size, k))
        self.batch_size = batch_size
        nn.init.xavier_normal_(self.diff.data)
        nn.init.xavier_normal_(self.disc.data)
        nn.init.xavier_normal_(self.theta.data)



    def forward(self, u, i, s):
        mult_result = torch.matmul(self.disc[i], self.theta[u].t())
        x = torch.mul(torch.eye(mult_result.shape[0]).cuda(), mult_result).sum(dim=1)
        x = torch.unsqueeze(x,1) + self.diff[i]
        pred = 1 / (1 + torch.exp(-x))
        real = torch.unsqueeze(s.float(),1).cuda()
        loss = F.binary_cross_entropy(pred, real)
        return loss


    def recommend(self, u):
        """Return recommended item list given users.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]

        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """
        pred = self.diff * (self.theta[u] - self.diff)
        return pred