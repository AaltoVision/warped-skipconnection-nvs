import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=0.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        if mask is None:
            return ssim_map.mean(1).mean(1).mean(1)
        else:
            return (ssim_map.mean(1)[mask]).mean()

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, sigma=0.5):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel, sigma)

    def forward(self, img1, img2):
        img1 = img1.mean(1).unsqueeze(1)
        img2 = img2.mean(1).unsqueeze(1)
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True, mask=None, sigma=0.5):
    img1 = img1.mean(1).unsqueeze(1)
    img2 = img2.mean(1).unsqueeze(1)

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel, sigma)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average,mask=mask)



def cal_grad2_error(flo, image, beta=1.0):
    """
    Calculate the image-edge-aware second-order smoothness loss 
    """

    def gradient(pred):
        D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy


    img_grad_x, img_grad_y = gradient(image)
    weights_x = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
    weights_y = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_y), 1, keepdim=True))

    #if flo.ndimension() == 3:
    #    flo = torch.unsqueeze(flo,1)

    dx, dy = gradient(flo)

    return (torch.mean(beta*weights_x*torch.abs(dx)) + torch.mean(beta*weights_y*torch.abs(dy))) / 2.0


def spL1(pred, gt):
    diff = torch.abs(pred - gt)
    return diff.type(torch.bool).float()