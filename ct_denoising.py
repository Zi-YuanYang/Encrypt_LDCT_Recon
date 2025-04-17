
from ipcl_python import PaillierKeypair, context, hybridControl, hybridMode,PaillierEncryptedNumber
import pickle as pkl
import numpy as np
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

from utils import *

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
    haha = np.random.random((96,8,8))
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

def generate_result():
    # generate Paillier scheme key pair
    pk, sk = PaillierKeypair.generate_keypair(1024)
    # Acquire QAT engine control
    context.initializeContext("QAT")

    # network load
    net = RED_CNN()
    net.load_state_dict(torch.load('./example/checkpoint/epoch_99_loss_0.008091.pth'))

    # data preparation
    path = './example/RED/data_0036.mat'
    img_data = scio.loadmat(path)['data'][128:128+64,128:128+64]
    img_data = np.expand_dims(img_data, axis=0)
    img_data = np.random.random((1,2,2))
    ori_shape = img_data.shape

    # validate in torch
    input_data = torch.FloatTensor(img_data).unsqueeze_(0)
    print("start to calc using pytorch")
    out = net(input_data).cpu().detach().numpy()
    del input_data

    #validate in encrypted 
    #Estimated completion time is 900 times the time of the first layer
    ct_img_data = encrypt_matrix(img_data,pk)
    # ct_img_data = img_data
    print("start to calc using encrypt")
    res = net.forward_encrypt(ct_img_data,encrypt_conv)
    print("calc using encrypt finished")

    res = decrypt_matrix(res,sk)
    print("result the same using two ways:",np.allclose(out,res))
    print("result the same using two ways:",np.allclose(out,res,atol=1e-5))
    print("result the same using two ways:",np.allclose(out,res,rtol=1e-3))
    res.resize(ori_shape)
    # save_plts(1,2,os.path.join("./result.jpg"),res,out,gray=True)
    scio.savemat("./result_128_128.mat", {'dec_data': res,'ori_data': img_data,'normal_data': out})


    context.terminateContext()

if __name__ == "__main__":
    
    # quick_validation()
    generate_result()