import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import dgl
import random
import numpy as np
import MMD

class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(42)

class AdversarialGAT(nn.Module):
    def __init__(self, vocab_size, embed_dim = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.gcn = GraphConv(embed_dim, 100, allow_zero_in_degree=True)

        self.fc = nn.Linear(100, 32)

        self.domain_clf = nn.Sequential(
            nn.Linear(128, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )

        self.defect_clf = nn.Sequential(
            nn.Linear(32, 1),
            # nn.Linear(32, 16),
            # nn.Tanh(),
            # nn.Linear(16, 1),
            nn.Sigmoid()
        )
      
    def max_process_graph(self, g, x):
        x = self.gcn(g, x)
        x = torch.relu(x)
        g.ndata['h'] = x
        return dgl.max_nodes(g, 'h')

    def forward(self, g, g_src, g_tar, alpha=1.0, mask=None, compute_node_mmd=True):
        x = self.embed(g.ndata['feat'].long())
        x_src = self.embed(g_src.ndata['feat'].long())
        x_tar = self.embed(g_tar.ndata['feat'].long())
      
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x_max = self.max_process_graph(g, x)
        x_src = self.max_process_graph(g_src, x_src)
        x_tar = self.max_process_graph(g_tar, x_tar)

        x_batch_mmd = self.fc(x_max)
        x_src_mmd = self.fc(x_src)
        x_tar_mmd = self.fc(x_tar)

        defect_prob = self.defect_clf(x_batch_mmd)
      
        reversed_feature = GradientReverse.apply(x_batch_mmd, alpha)

        domain_logits = self.domain_clf(reversed_feature)

        return x_batch_mmd, defect_prob, x_src_mmd, x_tar_mmd, x_loss_mmd_node

    if type_count == 0:
        return torch.tensor(0.0, device=src_feats.device)
    else:
        return total_mmd / type_count
