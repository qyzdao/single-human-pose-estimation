import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_memlab import profile, MemReporter

from utils import *


class ResiBlock(nn.Module):
    def __init__(self, in_features, filters, shortcut_conv=False):
        super(ResiBlock, self).__init__()

        self.filter1, self.filter2, self.filter3 = filters
        self.in_features = in_features
        self.shortcut_conv = shortcut_conv

        self.conv_block1 = nn.Sequential(
            nn.BatchNorm2d(self.in_features),
            nn.LeakyReLU(),
            nn.Conv2d(self.in_features, self.filter1, kernel_size=1, bias=False),
        )

        self.conv_block2 = nn.Sequential(
            nn.BatchNorm2d(self.filter1),
            nn.LeakyReLU(),
            nn.Conv2d(self.filter1, self.filter2, kernel_size=3, padding=(1,1), bias=False),
        )

        self.conv_block3 = nn.Sequential(
            nn.BatchNorm2d(self.filter2),
            nn.LeakyReLU(),
            nn.Conv2d(self.filter2, self.filter3, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Conv2d(self.in_features, self.filter3, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        if self.shortcut_conv:
            shortcut = self.shortcut(residual)
            x = torch.add(input=x, other=shortcut)
        else:
            x = torch.add(input=x, other=residual)
        return x


class UpResiBlock(nn.Module):
    def __init__(self, in_features, filters, stride=2):
        super(UpResiBlock, self).__init__()

        self.in_features = in_features
        self.filter1, self.filter2, self.filter3 = filters
        self.stride = stride

        r1 = self.stride * self.stride * self.filter2
        r2 = self.stride * self.stride * self.filter3

        self.conv_block1 = nn.Sequential(
            nn.BatchNorm2d(self.in_features),
            nn.LeakyReLU(),
            nn.Conv2d(self.in_features, self.filter1, kernel_size=1, bias=False)
        )

        self.conv_block2 = nn.Sequential(
            nn.BatchNorm2d(self.filter1),
            nn.LeakyReLU(),
            SubPixelConv2D(self.filter1, r1, kernel_size=3, padding=(1,1), upscale=self.stride),
        )

        self.conv_block3 = nn.Sequential(
            nn.BatchNorm2d(self.filter2),
            nn.LeakyReLU(),
            nn.Conv2d(self.filter2, self.filter3, kernel_size=1, bias=False),
        )

        self.shortcut = SubPixelConv2D(self.in_features, r2, kernel_size=1, padding=0, upscale=self.stride)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        shortcut = self.shortcut(residual)
        x = torch.add(input=x, other=shortcut)
        x = self.dropout(x)
        return x


class DownResiBlock(nn.Module):
    def __init__(self, in_features, filters, stride=2):
        super(DownResiBlock, self).__init__()

        self.in_features = in_features
        self.filter1, self.filter2, self.filter3 = filters
        self.stride = stride

        self.conv_block1 = nn.Sequential(
            nn.BatchNorm2d(self.in_features),
            nn.LeakyReLU(),
            nn.Conv2d(self.in_features, self.filter1, kernel_size=1, bias=False),
        )

        self.conv_block2 = nn.Sequential(
            nn.BatchNorm2d(self.filter1),
            nn.LeakyReLU(),
            nn.Conv2d(self.filter1, self.filter2, kernel_size=3, stride=self.stride, padding=(1,1), bias=False),
        )

        self.conv_block3 = nn.Sequential(
            nn.BatchNorm2d(self.filter2),
            nn.LeakyReLU(),
            nn.Conv2d(self.filter2, self.filter3, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Conv2d(self.in_features, self.filter3, kernel_size=1, stride=self.stride, bias=False)

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        shortcut = self.shortcut(residual)
        x = torch.add(input=x, other=shortcut)
        return x


class SEblock(nn.Module):
    def __init__(self, in_features, out_features, ratio=2):
        super(SEblock, self).__init__()
        self.ratio = ratio
        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Linear(self.out_features, self.out_features // ratio, bias=False)
        self.fc2 = nn.Linear(self.out_features // ratio, self.out_features, bias=False)
        self.conv1 = nn.Conv2d(self.in_features, 1, kernel_size=1, bias=False)

    def forward(self, l, h):
        mean = GlobalAvgPooling(l)
        cse = F.relu(self.fc1(mean))
        cse = torch.sigmoid(self.fc2(cse))
        b, c = cse.shape
        _, _, h, w = h.shape
        cse_repeat = cse.repeat(1, h*w).view(b,c,h,w)
        cse = torch.mul(cse_repeat, h)

        sse = torch.sigmoid(self.conv1(l))
        sse = sse.repeat(1, c, 1, 1)
        sse = torch.mul(sse, h)

        x = torch.add(input=cse, other=sse)
        return x


class film(nn.Module):
    def __init__(self, in_features, filters):
        super(film, self).__init__()
        self.in_features = in_features
        self.filters = filters

        self.fc1 = nn.Linear(self.in_features, self.filters)
        self.fc2 = nn.Linear(self.in_features, self.filters)

    def forward(self, cond, input_tensor):
        gamma = self.fc1(cond)
        beta = self.fc2(cond)

        b, c = gamma.shape
        _, _, h, w = input_tensor.shape
        gamma_repeat = gamma.repeat(1, h*w).view(b,c,h,w)
        beta_repeat = beta.repeat(1, h*w).view(b,c,h,w)

        mul = torch.mul(gamma_repeat, input_tensor)
        x = torch.add(input=mul, other=beta_repeat)
        return x

class film_ResiBlock(nn.Module):
    def __init__(self, in_features, filters):
        super(film_ResiBlock, self).__init__()

        self.in_features = in_features
        self.filters = filters

        self.conv1 = nn.Conv2d(in_features, filters, kernel_size=1, bias=False)
        self.activation = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=(1,1), bias=2)
        self.bn = nn.BatchNorm2d(filters, affine=False)
        self.film = film(128, filters)

    def forward(self, cond, input_tensor):
        a = self.conv1(input_tensor)
        a = self.activation(a)

        b = self.conv2(a)
        b = self.bn(b)
        b = self.film(cond, b)
        b = self.activation(b)

        return torch.add(input=a, other=b)

class SelfAttention(nn.Module):
    """
    Reference: https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    """
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X (W*H)xC//8
        proj_key = self.key_conv(x).view(m_batchsize,-1,width*height) # B X C//8 x (*W*H)
        energy = torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1)) # B,C,N
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma * out + x

        return out



class PixelWiseConcat(nn.Module):
    """
    Input:
    Lighting direction: (b, c1)
    Image: (b, c2, h, w)

    Expand lighting direction size to (b, c1, h, w), concat with image.

    Output size: (b, c1+c2, h, w)
    """
    def __init__(self):
        super(PixelWiseConcat, self).__init__()

    def forward(self, image, cond):
        b, c1 = cond.shape
        _, _, h, w = image.shape

        repeat_cond = cond.repeat(1, h*w).view(b, c1, h, w)
        new_tensor = torch.cat((repeat_cond, image), 1)

        return new_tensor


class SubPixelConv2D(nn.Module):
    """
    Sub Pixel Convolutional
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, upscale):
        super(SubPixelConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.upscale = upscale

        self.conv2d = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                        kernel_size=self.kernel_size, stride=1,
                        padding=self.padding, bias=False)

        self.subpixel = nn.PixelShuffle(self.upscale)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.subpixel(x)

        return x

def GlobalAvgPooling(x):
    """
    Input: (batch, channels, height, weight)
    Output: mean, (batch, channels)
    """
    b, c, h, w = x.shape
    x = x.view(b, c, h*w)
    mean = torch.mean(x, dim=2)

    return mean

"""
Coord Conv reference: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
"""
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size=1)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret
