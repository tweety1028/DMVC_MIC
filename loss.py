import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import sys

class MetaComprehensiveRegularization(nn.Module):
    def __init__(self):
        super(MetaComprehensiveRegularization, self).__init__()

    def forward_common(self, zc):

        loss_mcr = self.mutual_information_loss(zc, zc.t())
        return  loss_mcr
    def forward_special(self, zs, zc):
        loss_sp = 0
        for v in range(len(zs)):
            loss_sp+=self.mutual_information_loss(F.normalize(zc, dim=1),F.normalize(zs[v].t(), dim=0))
        return loss_sp
    def mutual_information_loss(self, z1, z2):

        z_hat = z1 @ z2

        mutual_info_lower_bound = -torch.diag(z_hat).sum()

        return mutual_info_lower_bound


def shannon_entropy(z_c):

    z = F.normalize(z_c, dim=1)

    matrix = z @ z.t()

    min_val = matrix.min()

    if min_val <= 0:
        matrix = matrix + abs(min_val) + 1e-10

    normalized_matrix = F.normalize(matrix, dim=1)

    # Compute the entropy.
    entropy_per_column = -torch.sum(normalized_matrix * torch.log(normalized_matrix), dim=1)



    return entropy_per_column.mean()