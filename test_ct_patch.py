
from ipcl_python import PaillierKeypair, context, hybridControl, hybridMode,PaillierEncryptedNumber
import pickle as pkl
import numpy as np
# from RED.REDCNN_old import *
from RED.REDCNN import *
from collections import OrderedDict
import torch
import scipy.io as scio
import torch.nn.functional as F
from matplotlib import image as image, pyplot as plt
import math
import time
import copy
import os
import multiprocessing as mp
mp.set_start_method("spawn",force=True)
import cv2 as cv
import pydicom
from glob import glob

from torch.autograd import Variable
from math import exp
import random
from utils import *
import argparse
def img2col(x, ksize, stride):
    c,h,w = x.shape
    img_col = []
    for i in range(0, h-ksize+1, stride):
        for j in range(0, w-ksize+1, stride):
            col = x[:, i:i+ksize, j:j+ksize].reshape(-1) 
            img_col.append(col)
    return np.array(img_col)

# @profile
def matrix_multiplication_for_conv2d(input:np.ndarray, kernel:np.ndarray,bias = 0, stride=1, padding=0,multiProcess = None):
    if padding > 0:
        input = padding_operation(input, padding)
    channel, input_h , input_w = input.shape
    out_channel, in_channel,  kernel_h, kernel_w = kernel.shape

    kernel = kernel.reshape(out_channel, -1)
    output_h = (math.floor((input_h - kernel_h) / stride) + 1)
    output_w = (math.floor((input_w - kernel_w) / stride) + 1)
    # output = [[[0.0 for i in range(output_w)] for i in range(output_h)] for i in range(out_channel)]
    output_shape = (out_channel,output_h,output_w)
    input = img2col(input,kernel_h,stride)
    if multiProcess == None:
        out = np.dot(kernel, input.T)+bias.reshape(-1,1)
    else:
        out = pardot(kernel, input.T,multiProcess[0],multiProcess[1])+bias.reshape(-1,1)
    output = np.reshape(out, output_shape) 
    return output

def encrypt_conv(x,module):
    in_channels = module.in_channels
    out_channels = module.out_channels
    padding,_ = module.padding
    stride,_ = module.stride
    kernel = module.weight.cpu().detach().numpy()
    bias = module.bias.cpu().detach().numpy()
    print(in_channels,out_channels,padding,stride,kernel.shape)
    x = matrix_multiplication_for_conv2d(x,kernel,bias,stride,padding,multiProcess=(4,4))
    # x = matrix_multiplication_for_conv2d(x,kernel,bias,stride,padding,multiProcess=None)
    return x

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (nrows, ncols, n, m) where
    n * nrows, m * ncols = arr.shape.
    This should be a view of the original array.
    """
    h, w = arr.shape
    n, m = h // nrows, w // ncols
    return arr.reshape(nrows, n, ncols, m).swapaxes(1, 2)

def do_dot(a,b,q):
    res = np.dot(a, b)
    q.send(res)

def pardot(a, b, nblocks, mblocks, dot_func=do_dot):
    """
    Return the matrix product a * b.
    The product is split into nblocks * mblocks partitions that are performed
    in parallel threads.
    """
    nblocks = min(nblocks,a.shape[0])
    mblocks = min(mblocks,b.shape[1])
    n_jobs = nblocks * mblocks
    print('running {} jobs in parallel'.format(n_jobs))
    a_blocks = blockshaped(a, nblocks, 1)
    b_blocks = blockshaped(b, 1, mblocks)
    out_blocks = np.array([[PaillierEncryptedNumber(None,None,None,None) for j in range(b.shape[1])] for i in range(a.shape[0])])
    threads = []
    parent_conns = []
    for i in range(nblocks):
        for j in range(mblocks): 
            parent_conn, child_conn = mp.Pipe() #huge amount of data, if use mp.Queue will make jam and deadlock
            th = mp.Process(target=dot_func,
            args=(
                a_blocks[i, 0, :, :],
                b_blocks[0, j, :, :],
                child_conn,
            ))
            th.start()
            threads.append(th)
            parent_conns.append(parent_conn)

    h,w = a.shape[0]//nblocks, b.shape[1]//mblocks
    for i in range(nblocks):
        for j in range(mblocks):
            out_blocks[i*h:i*h+h,j*w:j*w+w] = parent_conns[i*mblocks+j].recv()

    for th in threads:
        th.join()

    # out = out.reshape((a.shape[0],b.shape[1]))
    return out_blocks

def quick_validation():
    # generate Paillier scheme key pair
    pk, sk = PaillierKeypair.generate_keypair(1024)
    # Acquire QAT engine control
    context.initializeContext("QAT")
    print("start")
    haha = np.random.random((1,2,2))
    a = np.random.random((1,2,2))
    enc_haha = encrypt_matrix(haha,pk)
    rr = a * enc_haha
    out = decrypt_matrix(rr,sk)
    out2 = haha * a
    print(np.allclose(out,out2,atol=1e-5))
    kernel = np.random.random((96,96,5,5))
    enc_haha = encrypt_matrix(haha,pk)
    res = matrix_multiplication_for_conv2d(enc_haha,kernel,0,1,2,multiProcess=(4,4))
    print("finish")
    res = decrypt_matrix(res,sk)
    
    haha = torch.tensor(haha)
    haha = haha.unsqueeze(0)
    kernel = torch.tensor(kernel)
    res2 = F.conv2d(haha,kernel,padding=2).squeeze(0).numpy()
    print(np.allclose(res,res2,atol=1e-5))
    print(np.allclose(res,res2,atol=1e-8))
    print(np.allclose(res,res2,rtol=1e-3))

    context.terminateContext()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window.cuda()

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def ssim_met(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    return _ssim(img1, img2, window, window_size, channel, size_average)

def calculate_psnr(tensor1, tensor2):
    mse = F.mse_loss(tensor1, tensor2, reduction='none')
    mse = mse.mean(dim=[1, 2, 3])  # Compute mean over all but the first dimension
    max_pixel_value = 1.0  # assuming image pixel values are normalized to the range [0, 1]
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr


def val_img(out, lab):

    # model = model.cuda()
    # torch.cuda.empty_cache()
    # model.eval()
    # torch.cuda.empty_cache()
    # # model = model.cuda()
    # # print(fe.shape)
    # # print(opt.backbone)
    # with torch.no_grad():

    #         out = model(inp.cuda())

    channel, _, _, _ = out.shape

    ssim = []

    # print('1')
    psnrs = calculate_psnr(out.cuda(),lab.cuda())
    psnr = psnrs.squeeze().tolist()
    for c in range(channel):

        denoised = out[c,:,:,:].unsqueeze_(0)
        label = lab[c,:,:,:].unsqueeze_(0)
        ssim_temp = ssim_met(denoised.cuda(),label.cuda()).cpu().detach().numpy()
        ssim.append(ssim_temp)

        # denoised = out[c, 0, :, :].cpu().detach().numpy()
        # label = lab[c, 0, :, :].detach().numpy()
        # psnr_temp = peak_signal_noise_ratio(label,denoised)
        # ssim_temp = structural_similarity(denoised,label,data_range=1)
        # psnr.append(psnr_temp)
        del denoised, label
        # print("psnr: %f, ssim: %f" % (psnr_temp, ssim_temp))
    # del out, inp, lab, fe,ana
    torch.cuda.empty_cache()
    # model.cuda()
    
    return psnr,ssim



def generate_result(ind):
    # generate Paillier scheme key pair
    pk, sk = PaillierKeypair.generate_keypair(2048)
    # Acquire QAT engine control
    context.initializeContext("QAT")

    # network load
    net = RED_CNN()
    # net = RED_CNN_all_relu()
    net.load_state_dict(torch.load('./example/checkpoint/epoch_99_3layer.pth'))

    # data preparation
    # path = './example/RED/data_1927.mat'

    base_dir = './example/data/input/'
    thick_ = '3mm'
    volume_ = ['L109']
    print(volume_)
# './example/data/input/L067/quarter_3mm/*.IMA'

    ld = sorted(glob(base_dir + volume_[0] + '/quarter_' + thick_ + '/*.IMA'))
    nd = sorted(glob(base_dir + volume_[0] + '/full_' + thick_ + '/*.IMA'))
     
    ldct = pydicom.read_file(ld[ind]).pixel_array.astype(np.float32)
    ndct = pydicom.read_file(nd[ind]).pixel_array.astype(np.float32)


    ldct = (ldct + 1024.0) / 3072.0
    ndct = (ndct + 1024.0) / 3072.0

    ldct = torch.from_numpy(ldct).unsqueeze(0).unsqueeze(0)
    ndct = torch.from_numpy(ndct).unsqueeze(0).unsqueeze(0)



    ldct = F.interpolate(ldct, size=(256, 256), mode='bilinear', align_corners=False)
    ndct = F.interpolate(ndct, size=(256, 256), mode='bilinear', align_corners=False)

    x_start = random.randint(16,240)
    y_start = random.randint(16,240)
    patch_size = 64
    ldct_patch = ldct[:, :, x_start:x_start+patch_size, y_start:y_start+patch_size]
    ndct_patch = ndct[:, :, x_start:x_start+patch_size, y_start:y_start+patch_size]
    
    

    out = net(ldct_patch)
    pl_psnr, pl_ssim = val_img(out,ndct_patch)
    
    print('normal psnr: {}, normal ssim: {}'.format(pl_psnr,pl_ssim[0]))

    ldct_path_enc = ldct_patch.cpu().detach().numpy().squeeze(0).astype(np.float64)
    ct_img_data = encrypt_matrix(ldct_path_enc,pk)
    
    res = net.forward_encrypt(ct_img_data,encrypt_conv,sk,pk)
    res = torch.from_numpy(res.astype(np.float32)).unsqueeze(0)
    el_psnr, el_ssim = val_img(res.cuda(),ndct_patch)
    print('encrypted psnr: {}, encrypted ssim: {}'.format(el_psnr,el_ssim[0]))

    # res = decrypt_matrix(res,sk)


    # # img_data = scio.loadmat(path)['data'][128:128+64,128:128+64]
    # # img_data = np.expand_dims(img_data, axis=0)
    # # img_data = np.random.random((1,2,2))
    # # ori_shape = img_data.shape

    # # validate in torch
    # input_data = torch.FloatTensor(img_data).unsqueeze_(0)
    # print("start to calc using pytorch")
    # out = net(input_data).cpu().detach().numpy()
    # del input_data

    # #validate in encrypted 
    # #Estimated completion time is 900 times the time of the first layer
    # ct_img_data = encrypt_matrix(img_data,pk)
    # # ct_img_data = img_data
    # print("start to calc using encrypt")
    # res = net.forward_encrypt(ct_img_data,encrypt_conv)
    # print("calc using encrypt finished")

    # res = decrypt_matrix(res,sk)
    # print("result the same using two ways:",np.allclose(out,res))
    # print("result the same using two ways:",np.allclose(out,res,atol=1e-5))
    # print("result the same using two ways:",np.allclose(out,res,rtol=1e-3))
    # res.resize(ori_shape)
    # # save_plts(1,2,os.path.join("./result.jpg"),res,out,gray=True)
    # scio.savemat("./result_128_128.mat", {'dec_data': res,'ori_data': img_data,'normal_data': out})


    # context.terminateContext()

if __name__ == "__main__":
    # 
    # quick_validation()

    parser = argparse.ArgumentParser()

    parser.add_argument('--index', type=int, default=1)

    opt = parser.parse_args()
    generate_result(opt.index)