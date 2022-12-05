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

class BasicResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.Sequential(
            BasicBlock(1, 32),
            nn.Dropout(0.1),
            BasicBlock(32, 32),
            nn.Dropout(0.2),
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
            BasicBlock(1, 32),
            ChannelGate(32, 32, [ 'max' ]), #'avg'
            # SelfAttn(32),
            nn.Dropout(0.1),
            BasicBlock(32, 32),
            ChannelGate(32, 32, [ 'max' ]),
            # SelfAttn(32),
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