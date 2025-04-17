import matplotlib.pyplot as plt
import builtins
import datetime
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity 
import torch.nn as nn

old_print = builtins.print
def hook_print():
    def my_print(*args, **kwargs):
        old_print(datetime.datetime.now(), end=" ")
        old_print(*args, **kwargs)
    builtins.print = my_print
def unhook_print():
    builtins.print = old_print
hook_print()

# usage: save_plts(2,2,os.path.join(args.path,"./result.jpg"),img1,img2,img3,img4)
def save_plts(row,col,fileName,*args,gray=False,**kwargs):
    plt.figure(dpi=300,figsize=(12,8))
    for i,item in enumerate(args):
        plt.subplot(row,col,i+1)
        if gray == True:
            plt.imshow(item,'gray')
            continue
        plt.imshow(item)
    plt.savefig(fileName) 
    plt.close('all')

def padding_operation(input,padding):
    channel, input_h , input_w = input.shape
    for i in range(padding):
        input = np.insert(input,0,0,axis=1)
        input = np.insert(input,input_h+i+1,0,axis=1)
    for i in range(padding):
        input = np.insert(input,0,0,axis=2)
        input = np.insert(input,input_w+i+1,0,axis = 2)
    return input

# old version of implement of conv, less efficient
def matrix_multiplication_for_conv2d_old(input:np.ndarray, kernel:np.ndarray,bias = 0, stride=1, padding=0):
    if padding > 0:
        input = padding_operation(input, padding)
    # 计算输出大小
    channel, input_h , input_w = input.shape
    out_channel, in_channel,  kernel_h, kernel_w = kernel.shape

    output_h = (math.floor((input_h - kernel_h) / stride) + 1)
    output_w = (math.floor((input_w - kernel_w) / stride) + 1)

    output = [[[0.0 for i in range(output_w)] for i in range(output_h)] for i in range(out_channel)]
	
    # 不考虑性能
    for oc in range(out_channel):
        for ic in range(in_channel):
            for i in range(0, input_h - kernel_h + 1, stride):
                for j in range(0, input_w - kernel_w + 1, stride):
                    region = input[ic, i:i + kernel_h, j: j + kernel_w]
                    output[oc][int(i / stride)][int(j / stride)] += np.sum(region * kernel[oc][ic])
        # output[oc] += bias[oc]

    return np.array(output)

def encrypt_matrix(input,pk):
    output = []
    for batch in input:
        batch_output = []
        for data in batch:
            items = []
            for item in data:
                items.append(pk.encrypt(item.astype(float)))
                # items.append(pk.encrypt(item.astype(float)))
            batch_output.append(items)
        output.append(batch_output)
    output = np.array(output)
    return output

def decrypt_matrix(input,sk):
    output = []
    for batch in input:
        batch_output = []
        for data in batch:
            items = []
            for item in data:
                items.append(sk.decrypt(item))
            batch_output.append(items)
        output.append(batch_output)
    output = np.array(output)
    return output

def displaywin(img, low=0.42, high=0.62):
    img[img<low] = low
    img[img>high] = high
    img = (img - low)/(high - low) * 255
    return img

def get_psnr_ssim(out, lab):

    channel, _, _, _ = out.shape
    psnr = []
    ssim = []

    # print('1')
    for c in range(channel):

        denoised = out[c, 0, :, :]
        label = lab[c, 0, :, :]
        psnr_temp = peak_signal_noise_ratio(label,denoised,data_range=1)
        ssim_temp = structural_similarity(denoised,label,data_range=1)
        psnr.append(psnr_temp)
        ssim.append(ssim_temp)
        # print("psnr: %f, ssim: %f" % (psnr_temp, ssim_temp))
    return psnr,ssim

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)