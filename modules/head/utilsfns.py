import torch
import torch.nn as nn
import torch.nn.functional as F

def D(p, z):
    return - F.cosine_similarity(p, z.detach(), dim=-1, eps=1e-8).mean()

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if torch.distributed.is_initialized():
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor

@torch.no_grad()
def distributed_sinkhorn(out, iterations=3, epsilon=0.05):
    Q = torch.exp(out / epsilon).t()
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    B = Q.shape[1] * world_size # total batch size
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

def shoot_infs(inp_tensor):
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.min(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

def cross_entropy_shootinfs(pred, label):
    return F.nll_loss(shoot_infs(F.log_softmax(pred, dim=1)), label, None, None, -100, None, "none").mean()

class BufferQueue(nn.Module):
    def __init__(self, num_features, queue_size, l2norm=True):
        super().__init__()
        self.N = queue_size
        self.C = num_features
        self.l2norm = l2norm
        self.register_buffer("queue", torch.rand(self.N, self.C).cuda())

        if self.l2norm:
            self.queue = nn.functional.normalize(self.queue, dim=1)

    def forward(self, x):
        with torch.no_grad():
            bs = x.size(0)

            self.queue[bs:] = self.queue[:-bs].clone()
            self.queue[:bs] = x

            if self.l2norm:
                self.queue = nn.functional.normalize(self.queue, dim=1)

        return self.queue

    def extra_repr(self):
        return 'num_features={}, num_sample={}'.format(self.C, self.N)

class PCA(object):
    def __init__(self, dim=2, l2norm=False):
        self.n_components = dim
        self.l2norm = l2norm

    def fit(self, X):
        if self.l2norm:
            X = F.normalize(X, dim=1)
        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        covariance_matrix = covariance_matrix.float()
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj_mat = eigenvectors[:, 0:self.n_components]

    def transform(self, X):
        if self.l2norm:
            X = F.normalize(X, dim=1)
        X = X - self.mean
        out = X.matmul(self.proj_mat)

        if self.l2norm:
            out = F.normalize(out, dim=1)
        
        return out