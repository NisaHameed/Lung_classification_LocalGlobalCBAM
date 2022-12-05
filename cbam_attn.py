import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=8, pool_types=['avg','max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return 

import torch.nn as nn
import torch as T
from cbam import *

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
       
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
       
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class residual_block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(residual_block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv01 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # self.chatt=CBAM(32)


        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
       
         
        # self.chatt=CBAM(32)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv01(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out=self.chatt(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # out1 = self.conv01(x)
        # out1 = self.bn2(out1)
        # out1 = self.relu(out1)
         
        
        # out2=out1
        
        add=out+identity
        
        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out += identity
        out = self.relu(add)

        return out
    
class non_local_block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(non_local_block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
       
        self.chaatt = ChannelGate(32, 32, [ 'max' ])
        # self.chatt=CBAM(32)
        
        self.softmax = nn.Softmax(dim=-1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out1 = self.conv2(x)
        out2 = self.conv2(x)
        out3 = self.conv2(x)
        trans = torch.transpose(out1, 1, 2)
        
        mul= torch.mul(out2, trans)
        soft = self.softmax(mul) 
        
        # out = self.conv2(out3)
        # out = self.bn2(out)
        
        cha_att = self.chaatt(out3)
        mul2= torch.mul(soft, cha_att)
        
        
        
        add=mul2+x
        
        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out += identity
        # out = self.relu(add)

        return add
    
class BasicResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.Sequential(
            BasicBlock(1, 32),
            nn.Dropout(0.1),
            BasicBlock(32, 32),
            nn.Dropout(0.2),
            CBAM(32),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc1 = nn.Linear(32, 1)


    def forward(self, x):

        x = self.resnet(x)
        x = x.view(-1, x.size()[1])
        x = self.fc1(x)
        x = T.sigmoid(x)
        return x

# Copied from https://github.com/heykeetae/Self-Attention-GAN
class SelfAttn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(T.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = T.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = T.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x


        return out

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=8, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
    

class AllAtn(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnetAttn = nn.Sequential(
            conv3x3(1, 32),
            nn.ReLU(inplace=True),
            SelfAttn(32),
            nn.Dropout(0.1),
            SelfAttn(32),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc1 = nn.Linear(32, 1)


    def forward(self, x):

        x = self.resnetAttn(x)
        x = x.view(-1, x.size()[1])
        x = self.fc1(x)
        x = T.sigmoid(x)
        return x


class AllAtnBig(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnetAttn = nn.Sequential(
            conv3x3(1, 32),
            nn.ReLU(inplace=True),
            SelfAttn(32),
            nn.Dropout(0.1),
            SelfAttn(32),
            SelfAttn(32),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc1 = nn.Linear(32, 1)


    def forward(self, x):

        x = self.resnetAttn(x)
        x = x.view(-1, x.size()[1])
        x = self.fc1(x)
        x = T.sigmoid(x)
        return x

class LocalGlobalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnetAttn = nn.Sequential(
            BasicBlock(1,32),
            SelfAttn(32),
            nn.Dropout(0.1),
            # BasicBlock(32,32),
            # SelfAttn(32),
            CBAM(32),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.fc1 = nn.Linear(32, 1)


    def forward(self, x):

        x = self.resnetAttn(x)
        
        x = x.view(-1, x.size()[1])
        x = self.fc1(x)
        x = T.sigmoid(x)
        return x



class LocalGlobalNetwork_old(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnetAttn = nn.Sequential(
            
            BasicBlock(1,32),
            SelfAttn(32),
            nn.Dropout(0.1),
            BasicBlock(32,32),
            SelfAttn(32),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.fc1 = nn.Linear(32, 1)


    def forward(self, x):

        x = self.resnetAttn(x)
        
        x = x.view(-1, x.size()[1])
        x = self.fc1(x)
        x = T.sigmoid(x)
        return x