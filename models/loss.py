import torch
import torch.nn as nn
import numpy as np
from utils import pytorch_ssim

# GAN Loss
datype = torch.cuda.FloatTensor


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function with sigmoid input.
    z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,), all 0 or all 1.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def SquCE_loss(input, target):
    """
    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.   (x)
    - target: PyTorch Tensor of shape (N,), all 0 or all 1. (z)

    Returns:
    - The mean square of classification error over the minibatch of input data.
    z * (1-x)^2 +(1-z) * x^2
    """
    SquCE = target * (1-input)**2 + (1-target) * input**2
    return SquCE.mean()


def MAE_loss(input, target):
    """
     Inputs:
    - input: PyTorch Tensor of shape (N, C, H, W) .
    - target: PyTorch Tensor of shape (N, C, H, W).

    Returns:
    - A PyTorch Tensor containing the mean MAE loss 
    """
    MAE_fn = torch.nn.L1Loss(reduction='mean')
    return MAE_fn(input, target)


def FMAE_loss(input, target):
    """
     Inputs:
    - input: PyTorch Tensor of shape (N, C, H, W) .
    - target: PyTorch Tensor of shape (N, C, H, W).

    Returns:
    - MAE loss over the minibatch of the input images' spectrum.

    """
    MAE_fn = torch.nn.L1Loss(reduction='mean')

    Finput = torch.fft.fft2(input)
    Ftarget = torch.fft.fft2(target)
    Ferror = MAE_fn(torch.real(Finput), torch.real(Ftarget))
    return Ferror

    # # Better to normalize before L1 loss
    # N, C, _, _ = input.size()
    # Finput = torch.abs(torch.fft.fft2(input).view(N, C, -1))
    # Ftarget = torch.abs(torch.fft.fft2(target).view(N, C, -1))

    # MaxFinput = Finput.max(axis=2, keepdim=True)[0]
    # MaxFtarget = Ftarget.max(axis=2, keepdim=True)[0]
    # NormFinput = Finput / MaxFinput
    # NormFtarget = Ftarget / MaxFtarget

    # Ferror = MAE_fn(NormFinput, NormFtarget)
    # return Ferror


def MSE_loss(input, target):
    """
     Inputs:
    - input: PyTorch Tensor of shape (N, C, H, W) .
    - target: PyTorch Tensor of shape (N, C, H, W).

    Returns:
    - A PyTorch Tensor containing the mean MSE loss 
    """
    MSE_fn = torch.nn.MSELoss(reduction='mean')
    return MSE_fn(input, target)


def FMSE_loss(input, target):
    """
     Inputs:
    - input: PyTorch Tensor of shape (N, C, H, W) .
    - target: PyTorch Tensor of shape (N, C, H, W).

    Returns:
    - MSE loss over the minibatch of the DFT of input data.

    """
    MSE_fn = torch.nn.MSELoss(reduction='mean')

    Finput = torch.fft.fft2(input)
    Ftarget = torch.fft.fft2(target)
    Ferror = MSE_fn(torch.abs(Finput), torch.abs(Ftarget))
    return Ferror

    # # Better to normalize before L1 loss
    # Finput = torch.abs(torch.fft.fft2(input).view(N, C, -1))
    # Ftarget = torch.abs(torch.fft.fft2(target).view(N, C, -1))

    # MaxFinput = Finput.max(axis=2, keepdim=True)[0]
    # MaxFtarget = Ftarget.max(axis=2, keepdim=True)[0]
    # NormFinput = Finput / MaxFinput
    # NormFtarget = Ftarget / MaxFtarget

    # Ferror = MSE_fn(NormFinput, NormFtarget)
    # return Ferror


def SSIM_loss(input, target):
    """
     Inputs:
    - input: PyTorch Tensor of shape (N, C, H, W) .
    - target: PyTorch Tensor of shape (N, C, H, W).

    Returns:
    -  mean SSIM
    """
    SSIM_fn = pytorch_ssim.SSIM(size_average=False)
    ssim = (1+SSIM_fn(input, target))/2
    # ssim = 1-SSIM_fn(input, target)
    return -ssim.log().mean()

# def SSIM_loss(input, target):
#     """
#      Inputs:
#     - input: PyTorch Tensor of shape (N, C, H, W) .
#     - target: PyTorch Tensor of shape (N, C, H, W).

#     Returns:
#     -  mean SSIM
#     """
#     SSIM_fn = pytorch_ssim.SSIM(window_size=11)
#     ssim = 1-SSIM_fn(input, target)
#     return ssim


def PCC_loss(input, target):
    """
     Inputs:
    - input: PyTorch Tensor of shape (N, C, H, W) .
    - target: PyTorch Tensor of shape (N, C, H, W).

    Returns:
    -  mean PCC
    """
    x_mean, y_mean = torch.mean(input, dim=[2, 3], keepdim=True), torch.mean(
        target, dim=[2, 3], keepdim=True)
    vx, vy = (input-x_mean), (target-y_mean)
    sigma_xy = torch.mean(vx*vy, dim=[2, 3])
    sigma_x, sigma_y = torch.std(
        input, dim=[2, 3]), torch.std(target, dim=[2, 3])
    PCC = sigma_xy / ((sigma_x+1e-8) * (sigma_y+1e-8))
    return -PCC.log().mean()


def TV_loss(img):
    """
     Inputs:
    - img: PyTorch Tensor of shape (N, C, H, W) .

    Returns:
    - total variation of the img
    """
    [N, C, H, W] = img.size()
    TempBatch = torch.zeros([N, C, H+2, W+2]).type(datype)
    TV = 0
    for i in range(0, N):
        TempBatch[i, :, 1:H+1, 1:W+1] = img[i, :, :, :].clone()
        TV = TV + (TempBatch[i, :, 2:H+1, 1:W+1]-img[i, 0, 0:H-1, :]).pow(
            2).sum() + (TempBatch[i, :, 1:H+1, 2:W+1]-img[i, 0, :, 0:W-1]).pow(2).sum()
    return TV/N


def MAP_loss(img, target):
    [N, _, H, W] = img.size()
    MAPLoss = 0
    MSE_fn = torch.nn.MSELoss(reduction='mean')
    if torch.is_tensor(img) and torch.is_tensor(target):
        for i in range(N):
            x = torch.squeeze(img[i])
            y = torch.squeeze(target[i])
            x_max, y_max = torch.max(x, 0), torch.max(y, 0)
            loss = MSE_fn(x_max[0], y_max[0])
            MAPLoss = MAPLoss + loss
    return MAPLoss/N


def GradPenalty(D, xr, xf):
    """
    Gradient penalty for Discriminator of  Wasserstein GAN
    D: Discriminator model, xr: (N,C,H,W), xf:(N,C,H,W)
    torch.autograd.grad(), refer to:
    # https://blog.csdn.net/sinat_28731575/article/details/90342082
    # https://zhuanlan.zhihu.com/p/33378444
    # https://zhuanlan.zhihu.com/p/29923090
    """

    t = torch.randn(xr.size(0), 1, 1, 1).type(datype)
    xm = t*xr.clone() + (1-t)*xf.clone()
    xm.requires_grad_(True)
    WDmid = D(xm)
    # Compute the gradients of outputs w.r.t. the inputs.(same size as inputs)
    # grad_outputs: The “vector” in the Jacobian-vector product, usually all one. (same size as outputs)
    # create_graph: to equip GP with a grad_fn.
    # retain_graph: retain the graph used to compute the grad for the backward of GP
    Gradmid = torch.autograd.grad(outputs=WDmid, inputs=xm,
                                  grad_outputs=torch.ones_like(
                                      WDmid).type(datype),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
    Gradmid = Gradmid.view(Gradmid.size(0), -1)
    GP = torch.pow((Gradmid.norm(2, dim=1)-1), 2).mean()
    return GP


def Gloss_fn(loss_type, fake, real):

    # 10%-20% loss item othan than MAE loss
    if loss_type == 'L1-SSIM':
        Lloss = MAE_loss(fake, real)
        Adloss = SSIM_loss(fake, real)
        Gloss = Lloss + 2e-2*Adloss
    elif loss_type == 'L1-PCC':
        Lloss = MAE_loss(fake, real)
        Adloss = PCC_loss(fake, real)
        if torch.isnan(Adloss).item():
            Adloss = SSIM_loss(fake, real)
        Gloss = Lloss + 2e-2*Adloss
    elif loss_type == 'L1-FL1':
        Lloss = MAE_loss(fake, real)
        Adloss = FMAE_loss(fake, real)
        Gloss = Lloss + 1e-4 * Adloss
    elif loss_type == 'L1-TV':
        Lloss = MAE_loss(fake, real)
        Adloss = TV_loss(fake)
        Gloss = Lloss + 1e-4 * Adloss
    elif loss_type == 'L1':
        Lloss = MAE_loss(fake, real)
        Adloss = 0
        Gloss = Lloss
    elif loss_type == 'L2':
        Lloss = MSE_loss(fake, real)
        Adloss = 0
        Gloss = Lloss

    return Lloss, Adloss, Gloss


def clamp_weight(parameters, weight_clipping_limit):
    # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        p.data.clamp_(-weight_clipping_limit, weight_clipping_limit)


def clip_grad_value_(parameters, clip_value):
    """Clips gradient of an iterable of parameters at specified value.
    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.clamp_(min=-clip_value, max=clip_value)


def test_grad_value(parameters, optimizer):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    GradIter = filter(lambda p: p.grad is not None, parameters)
    GradList = list(filter(lambda p: torch.isnan(
        p.grad).any(), iter(parameters)))
    if not GradList:
        optimizer.step()
