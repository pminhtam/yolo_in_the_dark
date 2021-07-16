import torch
import torch.nn as nn
import torch.nn.functional as F




def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class AlginLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(AlginLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        y = F.pad(y,[1,1,1,1])
        diff0 = torch.abs(x-y[:,:,1:-1,1:-1])
        diff1 = torch.abs(x-y[:,:,0:-2,0:-2])
        diff2 = torch.abs(x-y[:,:,0:-2,1:-1])
        diff3 = torch.abs(x-y[:,:,0:-2,2:])
        diff4 = torch.abs(x-y[:,:,1:-1,0:-2])
        diff5 = torch.abs(x-y[:,:,1:-1,2:])
        diff6 = torch.abs(x-y[:,:,2:,0:-2])
        diff7 = torch.abs(x-y[:,:,2:,1:-1])
        diff8 = torch.abs(x-y[:,:,2:,2:])
        diff_cat = torch.stack([diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8])
        diff = torch.min(diff_cat,dim=0)[0]
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """

    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )


class LossBasic(nn.Module):
    """
    Basic loss function.
    """
    def __init__(self, gradient_L1=True):
        super(LossBasic, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.gradient = TensorGradient(gradient_L1)

    def forward(self, pred, ground_truth):
        return self.l2_loss(pred, ground_truth) + \
               self.l1_loss(self.gradient(pred), self.gradient(ground_truth))

class LossAnneal_i(nn.Module):
    """
    anneal loss function
    """
    def __init__(self, alpha=0.9998, beta=100):
        super(LossAnneal_i, self).__init__()
        self.global_step = 0
        # self.loss_func = LossBasic(gradient_L1=True)
        self.loss_func = CharbonnierLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, global_step, pred_i, ground_truth):
        """
        :param global_step: int
        :param pred_i: [batch_size, N, 3, height, width]
        :param ground_truth: [batch_size, 3, height, width]
        :return:
        """
        loss = 0
        for i in range(pred_i.size(1)):
            loss += self.loss_func(pred_i[:, i, ...], ground_truth[:,i,...])
        loss /= pred_i.size(1)
        loss = self.beta * self.alpha ** global_step * loss
        loss.requires_grad = True
        return loss


class BasicLoss(nn.Module):
    def __init__(self, eps=1e-3, alpha=0.998, beta=100):
        super(BasicLoss, self).__init__()
        self.charbonnier_loss = CharbonnierLoss(eps)
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, burst_pred, gt, gamma):
        b,N,c,h,w = burst_pred.size()
        burst_pred = burst_pred.view(b,c*N,h,w)
        burst_gt = torch.cat([gt[..., i::2, j::2] for i in range(2) for j in range(2)], dim=1)

        anneal_coeff = max(self.alpha ** gamma * self.beta, 1)

        burst_loss = anneal_coeff * (self.charbonnier_loss(burst_pred, burst_gt))

        single_loss = self.charbonnier_loss(pred, gt)

        loss = burst_loss + single_loss

        return loss, single_loss, burst_loss


if __name__ == "__main__":
    x = torch.rand((3,16,16))
    y = torch.rand((3,16,16))
    # y = x
    y = F.pad(y, [1, 1, 1, 1])
    print(y[:,1:-1, 1:-1].size())
    diff0 = torch.abs(x - y[:,1:-1, 1:-1])
    diff1 = torch.abs(x - y[:,0:-2, 0:-2])
    diff2 = torch.abs(x - y[:,0:-2, 1:-1])
    diff3 = torch.abs(x - y[:,0:-2, 2:])
    diff4 = torch.abs(x - y[:,1:-1, 0:-2])
    diff5 = torch.abs(x - y[:,1:-1, 2:])
    diff6 = torch.abs(x - y[:,2:, 0:-2])
    diff7 = torch.abs(x - y[:,2:, 1:-1])
    diff8 = torch.abs(x - y[:,2:, 2:])

    diff_cat = torch.stack([diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8])
    # print(diff0)
    # print(diff_cat.size())
    diff = torch.min(diff_cat, dim=0)
    print(diff[0].size())
    print(diff[0])