import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import cm
from matplotlib.pyplot import MultipleLocator
import matplotlib.image as mpimg
from pylab import subplots_adjust
import math
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy import signal, ndimage
import imageio
from scipy.ndimage.interpolation import zoom
import scipy.io
import h5py
from PIL import Image
from skimage import color, measure, morphology
import json
from math import fabs, sin, cos, radians, sqrt
import time
import sys
import os
from os import listdir
from os.path import join
import random
from pytorch_msssim import ssim, msssim, SSIM, MSSSIM
from torchvision import transforms as T


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def iter_batches(data_loader):
    while True:
        for i, (x, y) in enumerate(data_loader):
            yield (x, y)


# Show images during training
def show_images(images, path):
    images = np.squeeze(images.data.cpu().np())  # (N,H,W) ndarray
    (N, H, W) = images.shape
    # im mode, Ref to https://www.zhihu.com/question/68390406 DON'T use 'L' or 'P'
    JoImg = Image.new('F', (W*2, H*N//2))

    for i, im in enumerate(images):
        if i < 2:
            # box为2元数组（左上角坐标）或4元数组（左上右下）
            JoImg.paste(Image.fromarray(im), (i*W, 0, (i+1)*W, H))
        else:
            JoImg.paste(Image.fromarray(im),
                        ((i-N//2)*W, H, (i+1-N//2)*W, H*N//2))
    # JoImg = JoImg.convert("RGB")
    JoImg = np.asarray(JoImg)  # JoImg.save(path)
    plt.imsave(path, JoImg, cmap=cm.hot)


def normImg(image):  # , min, max
    "Non-lossy conversion from float image to uint8"
    img = image.astype(np.float64)
    minImg, maxImg = np.amin(img), np.amax(img)
    img = (img - minImg) / (maxImg - minImg)
    img = (255 * img).astype(np.uint8)
    return img


def saveImgfromTensor(images, path, iters):
    images = images.permute(0, 2, 3, 1)
    images = images.data.cpu().numpy()  # (N,H,W,C) ndarray

    for i, im in enumerate(images):
        imgpath = path + "%d_%d.png" % (iters, i)

        plt.imsave(imgpath, np.squeeze(images[i]), cmap=cm.hot)


def plot_lossCurve(result):

    recordLen = len(result['epoches'])
    train_loss, val_loss = result['train_loss'], result['val_loss']

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)

    ax1.yaxis.grid(True, linestyle='-')
    ax1.plot(result['epoches'],  train_loss, 'k-',  label="train", linewidth=2)
    ax1.plot(result['epoches'],  val_loss, 'r-', label="val", linewidth=2)
    ax1.set_xlabel('Epoches', fontsize=16)
    ax1.set_ylabel('Loss', fontsize=16)
    plt.legend(loc='upper right')

    plt.show()


def compute_mean(train_path, test_path):

    train_filenames = [join(train_path, x) for x in listdir(train_path)]
    test_filenames = [join(test_path, x) for x in listdir(test_path)]
    total_filenames = train_filenames + test_filenames
    imgNum = len(total_filenames)

    img = np.array(Image.open(total_filenames[0]))

    if img.ndim == 2:
        C = 1
    elif img.ndim == 3:
        C = img.shape[2]
    # compute the mean over each channel
    sumMean = []
    for c in range(C):
        sumMean.append(0)
        for index in range(imgNum):
            img = np.array(Image.open(total_filenames[index]))
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            sumMean[c] += np.sum(img[:, :, c])
    means = [float(elem)/(255*img.shape[0]*img.shape[1]*imgNum)
             for elem in sumMean]

    # compute the std over each channel
    sumSquare = []
    for c in range(C):
        sumSquare.append(0)
        for index in range(imgNum):
            img = np.array(Image.open(total_filenames[index]))
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            sumSquare[c] += np.sum((img[:, :,
                                        c].astype(np.float32)/255. - means[c])**2)
    stds = [sqrt(elem/(imgNum*img.shape[0]*img.shape[1]))
            for elem in sumSquare]
    return means, stds


# Image patch extraction
def extract_patch(img1, img2, rows, cols, patchH, patchW, overlap, path):
    mkdir(path+'x/')
    mkdir(path+'y/')
    mkdir(path+'ARimgPat')
    mkdir(path+'ORimgPat')
    img1, img2 = img1[0:rows, 0:cols], img2[0:rows, 0:cols]
    ImgMean1, ImgStd1 = np.mean(img1), np.std(img1, ddof=1)
    # , vmin=ImgMean1-ImgStd1, vmax=1
    plt.imsave(path+'AR.png', img1, cmap=cm.hot)
    np.save(path+'AR.npy', img1)
    ImgMean2, ImgStd2 = np.mean(img2), np.std(img2, ddof=1)
    # , vmin=ImgMean2-ImgStd2, vmax=1
    plt.imsave(path+'OR.png', img2, cmap=cm.hot)
    np.save(path+'OR.npy', img2)

    count = 0
    for i in range((rows-overlap)//(patchH-overlap)):
        for j in range((cols-overlap)//(patchW-overlap)):
            count += 1
            np.save(path+'x/'+str(count).zfill(2)+'.npy', img1[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                                               j*(patchW-overlap):(j+1)*patchW-j*overlap])
            np.save(path+'y/'+str(count).zfill(2)+'.npy', img2[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                                               j*(patchW-overlap):(j+1)*patchW-j*overlap])
            plt.imsave(path+'ARimgPat/'+str(count).zfill(2)+'.png', img1[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                                                         j*(patchW-overlap):(j+1)*patchW-j*overlap], cmap='hot')
            plt.imsave(path+'ORimgPat/'+str(count).zfill(2)+'.png', img2[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                                                         j*(patchW-overlap):(j+1)*patchW-j*overlap], cmap='hot')
    return


def extract_TestPatch(ARORStr, img, rows, cols, patchH, patchW, overlap, path):
    if ARORStr == 'AR':
        pathXY = path + 'x/'
        # pathPat = path + 'ARimgPat/'

    elif ARORStr == 'OR':
        pathXY = path + 'y/'

    mkdir(pathXY)
    # mkdir(pathPat)

    ImgMean, ImgStd = np.mean(img), np.std(img, ddof=1)
    plt.imsave(path + ARORStr + '.png', np.power(img, 1),
               cmap=cm.hot)  # , vmin=ImgMean-ImgStd, vmax=1
    # np.save(path + ARORStr + '.npy', img)

    count = 0
    for i in range(int(rows-overlap)//int(patchH-overlap)):
        for j in range(int(cols-overlap)//int(patchW-overlap)):
            count += 1
            np.save(pathXY+str(count).zfill(2)+'.npy', img[i*(patchH-overlap):(i+1)*patchH-i*overlap,
                                                           j*(patchW-overlap):(j+1)*patchW-j*overlap])
            # plt.imsave(pathPat+str(count).zfill(2)+'.png', img[i*(patchH-overlap):(i+1)*patchH-i*overlap,
            #                     j*(patchW-overlap):(j+1)*patchW-j*overlap], cmap='hot')
    return


def extract_TestimgP(ARORStr, img, rows, cols, patchH, patchW, overlap, path):
    if ARORStr == 'AR':
        pathXY = path + 'x/'

    elif ARORStr == 'OR':
        pathXY = path + 'y/'

    mkdir(pathXY)
    plt.imsave(path + ARORStr + '.png', np.power(img, 1), cmap=cm.hot)

    img = (img*255).astype(np.uint8)

    count = 0
    for i in range(int(rows-overlap)//int(patchH-overlap)):
        for j in range(int(cols-overlap)//int(patchW-overlap)):
            count += 1
            imageio.imwrite(pathXY+str(count).zfill(2)+'.png', img[i*(patchH-overlap):
                                                                   (i+1)*patchH-i*overlap, j*(patchW-overlap):(j+1)*patchW-j*overlap])
    return


def extract_TestimgP1(ARORStr, img, rows, cols, patchH, patchW, overlap, path):
    if ARORStr == 'AR':
        pathXY = path + 'x/'

    elif ARORStr == 'OR':
        pathXY = path + 'y/'

    mkdir(pathXY)
    plt.imsave(path + ARORStr + '.png', np.power(img, 1), cmap=cm.hot)

    count = 0
    for i in range(int(rows-overlap)//int(patchH-overlap)):
        for j in range(int(cols-overlap)//int(patchW-overlap)):
            count += 1
            np.save(pathXY+str(count).zfill(2)+'.npy', img[i*(patchH-overlap):
                                                           (i+1)*patchH-i*overlap, j*(patchW-overlap):(j+1)*patchW-j*overlap])
    return
# Image patch stitch


def stitch_patch_V(SR_path, patchs, rows, cols, patchH, patchW, overlap):
    '''
    overlapping same areas (or half) between neighboring blocks
    '''
    imgs = np.zeros([rows, cols])

    numW, numH = (rows-overlap)//(patchH-overlap), (cols -
                                                    overlap)//(patchW-overlap)
    count = 0
    for i in range(numW):
        for j in range(numH):
            patch = patchs[count]
            # imgs[i*(patchH-overlap):(i+1)*patchH-i*overlap,
            #           j*(patchW-overlap):(j+1)*patchW-j*overlap] = patch
            imgs[i*(patchH-overlap)+int((i > 0)/2*overlap):(i+1)*patchH-int((i+1/2-(i+1 == numH)/2)*overlap),
                 j*(patchW-overlap)+int((j > 0)/2*overlap):(j+1)*patchW-int((j+1/2-(j+1 == numW)/2)*overlap)] = \
                patch[int((i > 0)/2*overlap):int(patchH+((i+1 == numH)/2-1/2)*overlap), int((j > 0)/2*overlap):
                      int(patchW+((j+1 == numW)/2-1/2)*overlap)]
            # imageio.imwrite(patch_path + str(count+1).zfill(2) + '.tiff', patchs[count])
            count += 1

    imgs = imgs[0:(i+1)*patchH-i*overlap, 0:(j+1)*patchW-j*overlap]

    # self-normalization
    imgs[imgs < 0] = 0
    imgmax = np.amax(imgs)
    imgs = imgs/imgmax

    np.save(SR_path+'.npy', imgs)
    # scipy.io.savemat(SR_path+'SR.mat', {'SR':imgs})
    # np.power(imgs,1), cmap=cm.hot, vmin=0.1, vmax=0.6
    plt.imsave(SR_path+'.png', np.power(imgs, 1),
               cmap=cm.hot)  # , vmin=0.7*ImgMean
    return imgs


def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if not torch.is_tensor(img1):
        img1, img2 = torch.from_numpy(img1.astype(
            np.float32)), torch.from_numpy(img2.astype(np.float32))
    if img1.ndim == 2:
        img1 = torch.unsqueeze(torch.unsqueeze(img1, 0), 0)
        img2 = torch.unsqueeze(torch.unsqueeze(img2, 0), 0)
    ssims = ssim(img1, img2, val_range=1, size_average=False)  # return (N,)
    return ssims.mean().item()


def calculate_psnr(img1, img2):
    # To calculate PSNR between two images in ndarray
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    peak = np.amax(img2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(peak / math.sqrt(mse))


def calculate_pcc(x, y):
    '''
    To calculate PCC between two images(2D ndarray or 4D tensor) 
    x: distorted image, y: ground truth
    '''
    if x.ndim == 2:
        x_mean, y_mean = np.mean(x), np.mean(y)
        vx, vy = (x-x_mean), (y-y_mean)
        sigma_xy = np.mean(vx*vy)
        sigma_x, sigma_y = np.std(x, ddof=0), np.std(y, ddof=0)
        PCC = sigma_xy / ((sigma_x+1e-8) * (sigma_y+1e-8))
        return PCC.mean()
    elif x.ndim == 4:
        x_mean, y_mean = torch.mean(x, dim=[2, 3], keepdim=True), torch.mean(
            y, dim=[2, 3], keepdim=True)
        vx, vy = (x-x_mean), (y-y_mean)
        sigma_xy = torch.mean(vx*vy, dim=[2, 3])
        sigma_x, sigma_y = torch.std(x, dim=[2, 3]), torch.std(y, dim=[2, 3])
        PCC = sigma_xy / ((sigma_x+1e-8) * (sigma_y+1e-8))
        return PCC.mean().item()
    else:
        raise ValueError("ndim error!")


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        None


def progressbar(it, prefix="", size=30, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" %
                   (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 14

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(12, 8))

        ax[0].imshow(image,  cmap=cm.hot)
        ax[1].imshow(mask,  cmap=cm.hot)
    else:
        f, ax = plt.subplots(2, 2, figsize=(12, 8))

        ax[0, 0].imshow(original_image,  cmap=cm.hot)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask,  cmap=cm.hot)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image,  cmap=cm.hot)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask,  cmap=cm.hot)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)


def visualize_hist(image, mask, original_image=None, original_mask=None):
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(6, 4))

        ax[0].hist(image, facecolor="blue")
        ax[1].hist(mask, facecolor="blue")
    else:
        f, ax = plt.subplots(2, 2, figsize=(6, 4))

        ax[0, 0].hist(original_image, facecolor="blue")
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].hist(original_mask, facecolor="blue")
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].hist(image, facecolor="blue")
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].hist(mask, facecolor="blue")
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
