import argparse
import os
import random
import torch
import torch.nn as nn

import torch.nn.parallel
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet
import math
from torchsummary import summary
from collections import OrderedDict
#from lib.pointnet_util import farthest_point_sample, index_points, square_distance

##############################################origion posenet####################################
psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet34'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        x_r = x_v @ attention # b, c, n 
        x = x + x_r
        return x

class CB_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        y_t = y.permute(0, 2, 1) # b, n, c x= b, c, m        
        energy =  y_t @ x # b, n, m 
        attention = self.softmax(energy)# b, n, m 
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))# b, n, m 
        x_r = x @ attention.permute(0, 2, 1) # b, c, n 
        y_r = y @ attention # b, c, m
        x = x + y_r
        y = y + x_r

        out = torch.cat([x,y], dim=1)
        return out

class PEFeat(nn.Module):
    def __init__(self, num_points):
        super(PEFeat, self).__init__()
        self.combine1 = CB_Layer()
        self.combine2 = CB_Layer()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = self.combine1(x,emb)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = self.combine2(x,emb)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024 = 1408

class PPFeat(nn.Module):
    def __init__(self, num_points):
        super(PPFeat, self).__init__()
        self.combine1 = CB_Layer()
        self.combine2 = CB_Layer()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.m_conv1 = torch.nn.Conv1d(3, 64, 1)
        self.m_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
    def forward(self, x, model_points):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.m_conv1(model_points))
        pointfeat_1 = self.combine1(x,emb)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.m_conv2(emb))
        pointfeat_2 = self.combine2(x,emb)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024 = 1408

class P_P_EFeat(nn.Module):
    def __init__(self, num_points):
        super(P_P_EFeat, self).__init__()
        self.combine = CB_Layer()
        self.conv1 = torch.nn.Conv1d(1408, 1408, 1)
        self.conv2 = torch.nn.Conv1d(1408, 1408, 1)
        self.conv3 = torch.nn.Conv1d(2816, 2816, 1)
        self.conv4 = torch.nn.Conv1d(5632, 2816, 1)
        self.conv5 = torch.nn.Conv1d(2816, 1408, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
    def forward(self, PEfeat, PPfeat):
        PEfeat = F.relu(self.conv1(PEfeat))
        PPfeat = F.relu(self.conv2(PPfeat))

        PPEfeat = self.combine(PEfeat, PPfeat)#c = 2816
        PPEfeat = F.relu(self.conv3(PPEfeat))

        ap_x = self.ap1(PPEfeat)

        ap_x = ap_x.view(-1, 2816, 1).repeat(1, 1, self.num_points)
        PPEfeat = torch.cat([PPEfeat, ap_x], 1)#2816+2816
        PPEfeat = F.relu(self.conv4(PPEfeat))
        PPEfeat = F.relu(self.conv5(PPEfeat))

        return PPEfeat #1480


class PoseNet_trans(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet_trans, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat1 = PEFeat(num_points)
        self.feat2 = PPFeat(num_points)
        self.feat = P_P_EFeat(num_points)

        #self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        #self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        #self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        #self.conv4_r = torch.nn.Conv1d(128, num_obj * 4, 1)  # quaternion
        self.conv4_t = torch.nn.Conv1d(128, 3, 1)  # translation
        self.conv4_c = torch.nn.Conv1d(128, 1, 1)  # confidence

        self.conv1_pr = torch.nn.Linear(3, 64)
        self.conv2_pr = torch.nn.Linear(64, 128)
        self.sa_pr2 = SA_Layer(128)
        self.conv3_pr = torch.nn.Linear(128, 256)
        self.conv4_pr = torch.nn.Linear(256, 512)
        self.sa_pr4 = SA_Layer(512)
        self.conv5_pr = torch.nn.Linear(512, 1024)

        self.conv1_r = torch.nn.Conv1d(2432, 1024, 1)
        self.conv2_r = torch.nn.Conv1d(1024, 512, 1)
        self.conv3_r = torch.nn.Conv1d(512, 256, 1)
        self.conv4_r = torch.nn.Conv1d(256, 128, 1)
        self.conv5_r = torch.nn.Conv1d(128, 64, 1)
        self.conv6_r = torch.nn.Conv1d(64, 4, 1)
        self.ap = torch.nn.AvgPool1d(500)

        self.num_obj = num_obj

    def forward(self, img, x, model_points, choose, obj):
        out_img = self.cnn(img)

        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        x = x.transpose(2, 1).contiguous()
        model_points = model_points.transpose(2, 1).contiguous()
        ap_x1 = self.feat1(x, emb)
        ap_x2 = self.feat2(x, model_points)
        ap_x = self.feat(ap_x1, ap_x2)

        #rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))

        #rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        #rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        #rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, 1, self.num_points)


        x_trans = (x-(x+out_tx)).transpose(1,2)#(500*3)

        ap_w = self.ap(ap_x).repeat(1,1,500)
        #rx = torch.cat([ap_x, x_trans], 1)
        rx1 = F.relu(self.conv1_pr(x_trans))#(64*500)R
        #rx1 = self.sa_pr1(rx1.transpose(1,2).contiguous()).transpose(1,2).contiguous()
        rx2 = F.relu(self.conv2_pr(rx1))#(128*500)R
        rx2 = self.sa_pr2(rx2.transpose(1,2).contiguous()).transpose(1,2).contiguous()
        rx3 = F.relu(self.conv3_pr(rx2))#(256*500)R
        #rx3 = self.sa_pr3(rx3.transpose(1,2).contiguous()).transpose(1,2).contiguous()
        rx4 = F.relu(self.conv4_pr(rx3))#(512*500)R
        rx4 = self.sa_pr4(rx4.transpose(1,2).contiguous()).transpose(1,2).contiguous()
        rx5 = F.relu(self.conv5_pr(rx4)).transpose(1,2).contiguous()#(1024*500)
        #rx5 = self.sa_pr5(rx5)

        ry = torch.cat([rx5,ap_w],1)
        ry1 = F.relu(self.conv1_r(ry))#(1024*500)
        ry2 = F.relu(self.conv2_r(ry1))+rx4.transpose(1,2).contiguous()#(512*500)
        ry3 = F.relu(self.conv3_r(ry2))+rx3.transpose(1,2).contiguous()#(256*500)
        ry4 = F.relu(self.conv4_r(ry3))+rx2.transpose(1,2).contiguous()#(128*500)
        ry5 = F.relu(self.conv5_r(ry4))#(64*500)
        out_rx = self.conv6_r(ry5).view(bs, 4, self.num_points)



        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        return out_rx, out_tx, out_cx, emb.detach()


###################################################Refine#############################################


class PoseRefineNet_trans(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet_trans, self).__init__()
        self.num_points = num_points
        self.feat1 = PEFeat(num_points)
        self.feat2 = PPFeat(num_points)
        self.feat = P_P_EFeat(num_points)
        
        self.conv1_r = torch.nn.Linear(1408, 512)
        self.conv1_t = torch.nn.Linear(1408, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*4) #quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj*3) #translation

        self.num_obj = num_obj
        self.ap = torch.nn.AvgPool1d(num_points)

    def forward(self, x, emb, model_points, obj):
        bs = x.size()[0]
        
        x = x.transpose(2, 1).contiguous()
        model_points = model_points.transpose(2, 1).contiguous()
        ap_x1 = self.feat1(x, emb)
        ap_x2 = self.feat2(x, model_points)
        ap_x = self.feat(ap_x1, ap_x2)
        ap_x = self.ap(ap_x)
        ap_x = ap_x.view(-1, 1408)
        #print(ap_x.size())

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx)
        tx = self.conv3_t(tx)

        rx = rx.view(bs, self.num_obj, 4)
        tx = tx.view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx



