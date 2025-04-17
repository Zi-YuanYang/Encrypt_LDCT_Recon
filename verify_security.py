from ipcl_python import PaillierKeypair, context, hybridControl, hybridMode,PaillierEncryptedNumber
import pickle as pkl
import numpy as np
from RED.REDCNN import *
from collections import OrderedDict
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import image as image, pyplot as plt
from resnet import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import glob
# import torchvision.models as models
import math
import time
import copy
import os
from utils import *

class TrainDateset(Dataset):
    def __init__(self,datadir):
        super(TrainDateset, self).__init__()
        self.data = glob.glob(datadir+"/test/input/*")
        self.datadir = datadir
    def __getitem__(self,index):
        path = self.data[index]
        img_data = scio.loadmat(path)['data']
        img_data = np.expand_dims(img_data,0)
        # label_path = os.path.join(self.datadir,"test","label",os.path.basename(path))
        # label = scio.loadmat(label_path)['data']
        # return img_data,label
        return img_data
    def __len__(self):
        return len(self.data)

class AttackModel(nn.Module):
    def __init__(self, out_ch=96):
        super(AttackModel, self).__init__()
        self.conv1 = nn.Conv2d(96, out_ch, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(out_ch, 1, kernel_size=5, stride=1, padding=2, bias=True)

        self.relu = nn.ReLU()

    def forward(self, input1,input2):
        out = self.conv1(self.relu(input1))
        out = self.conv2(self.relu(out))
        out = out+input2
        out = self.relu(out)
        return out




if __name__ == "__main__":
    device = "cuda"
    batchs = 5000
    epochs = 50

    
    model_normal = AttackModel().to(device)
    # model_normal.apply(weight_init)
    model_enc = AttackModel().to(device)
    net = RED_CNN_all_relu()
    net.load_state_dict(torch.load('./example/checkpoint/best_param_all_relu.pth'))
    net = net.to(device)
    optimizer_normal = optim.Adam(model_normal.parameters(),1e-6)
    optimizer_enc = optim.Adam(model_enc.parameters(),1e-6)
    loss_fn = nn.MSELoss()

    # pk, sk = PaillierKeypair.generate_keypair(200)

    # dataset = TrainDateset("../data/geometry_1")
    # data_loader = DataLoader(dataset, batch_size=1,shuffle=False, num_workers = 4)

    # img = next(iter(data_loader)).to(device)
    path = '../data/geometry_1/train/input/data_0621.mat'
    img = scio.loadmat(path)['data']
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img).to(device)
    with torch.no_grad():
            # todo net 重写，确定返回哪些内容，输入和标签
        input1,input2,out = net.forward_return_mid(img)

    for batch_idx in range(batchs):
    
        pred_normal = model_normal(input1,input2)
        ct_input1 = torch.randn_like(input1)*input1
        ct_input2 = torch.randn_like(input2)*input2
        pred_crypted = model_enc(ct_input1,ct_input2)

        loss_normal = loss_fn(pred_normal,out)
        loss_crypted = loss_fn(pred_crypted,out)

        loss_normal.backward()
        loss_crypted.backward()

        optimizer_normal.step()
        optimizer_enc.step()

        if (batch_idx+1)%100 == 0:
            print('iter %03d, loss_normal %f, loss_enc %f' %  (batch_idx, loss_normal.item(), loss_crypted.item()))

    torch.save(model_normal.state_dict(),"attack_model_normal.pt")
    torch.save(model_enc.state_dict(),"attack_model_enc.pt")
    scio.savemat("./result_attack.mat", {"input":img.detach().squeeze(0).cpu().numpy().transpose(1,2,0),
                                        "raw_result":out.detach().squeeze(0).cpu().numpy().transpose(1,2,0),
                                        "attack_normal":pred_normal.detach().squeeze(0).cpu().numpy().transpose(1,2,0),
                                        "attach_enc":pred_crypted.detach().squeeze(0).cpu().numpy().transpose(1,2,0)})
    save_plts(2,2,"vs_.png",displaywin(img.detach().squeeze(0).cpu().numpy().transpose(1,2,0)),displaywin(out.detach().squeeze(0).cpu().numpy().transpose(1,2,0))
                            ,displaywin(pred_normal.detach().squeeze(0).cpu().numpy().transpose(1,2,0)),displaywin(pred_crypted.detach().squeeze(0).cpu().numpy().transpose(1,2,0)),gray=True)
 

            





