import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        # self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        # if self.relu is not None:
        #     x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        #torch.max(x, 1)[0] 表示对张量 x 沿着第一个维度（通常是行）进行最大值的计算。返回的结果是一个包含每行最大值的张量。
        #unsqueeze(1)是为了在第二维度上增加一个维度，使得张量的形状变为 (N, 1)，其中N是原始张量的行数。
        # torch.mean(x, 1) 表示对张量 x 沿着第一个维度进行平均值的计算。
        # 返回的结果是一个包含每行平均值的张量。unsqueeze(1) 的作用同上，将张量的形状变为 (N, 1)
        # 这段代码的作用是将张量 x 每行的最大值和平均值按列拼接成一个新的张量。
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        #z-pool
        x_compress = self.compress(x)
        #7*7的卷积
        x_out = self.spatial(x_compress)
        #batch Norm+sigmoid
        scale = torch.sigmoid_(x_out) 
        return x * scale
class CDA(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CDA, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        b, c, _, _ = x.size()
        #将通道维度c和高度维度h交换,
        x_perm1 = x.permute(0,2,1,3).contiguous()#200,64,5,5
        #对通道维度进行注意力操作
        x_1 = self.ChannelGateH(x_perm1)#200,5,640,5#HCW
        #第一个分支
        #恢复原始的维度顺序
        x_1 = x_1.permute(0,2,1,3).contiguous()#200,640,5,5#CHW
        #对输入张量x进行维度排列，将宽度维度w和通道维度c交换
        x_2 = x.permute(0,3,2,1).contiguous()#200,5,5,640#WHC
        x_2 = self.ChannelGateW(x_2)#200,5,5,640
        #恢复原始的维度顺序
        x_2 = x_2.permute(0,3,2,1).contiguous()#200,640,5,5#CHW
        x_3=self.SpatialGate(x)
        x_out11=x_1*x_2
        x_out22=x_2*x_3
        x_out33=x_1*x_3
        x_out = (1/3)*(x_out11+x_out22+x_out33)#200,640,5,  
        return x_out  