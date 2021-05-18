import torch
import torch.nn as nn
import torch.nn.functional as F


class IRT(nn.Module):
    def __init__(self, item_size, user_size, batch_size):
        super().__init__()
        self.diff = nn.Parameter(torch.empty(item_size, 1))
        self.disc = nn.Parameter(torch.empty(item_size, 1))
        self.theta = nn.Parameter(torch.empty(user_size, 1))
        self.batch_size = batch_size
        nn.init.xavier_normal_(self.diff.data)
        nn.init.xavier_normal_(self.disc.data)
        nn.init.xavier_normal_(self.theta.data)



    def forward(self, u, i, s):
        x = -(1.702 * self.disc[i] * (self.theta[u] - self.diff[i]))
        pred = 1 / (1 + torch.exp(x)).squeeze(-1)
        real = s.float().cuda()
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