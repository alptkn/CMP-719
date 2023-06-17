import torch
import torch.nn as nn
from torchvision import models
import torchvision
import torch.nn.functional as F


from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
import math




# Main Convolutional Block with RELU activation
class ConvMainBlock(nn.Module):
    def __init__(self, inp, out, kernel_size, stride=1, padding=0):
        super(ConvMainBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(inp, out, kernel_size, stride, padding ), nn.ReLU(True))

    def forward(self, input):
        x = self.conv(input)

        return x

#Transpose Convolution Block with RELU activation
class TransposeConv(nn.Module):
    def __init__(self, inp, out, kernel_size, stride = 1, padding = 0, out_padding = 0):
        super(TransposeConv, self).__init__()
        self.trans = nn.Sequential(nn.ConvTranspose2d(inp, out, kernel_size=kernel_size, stride = stride, padding=padding, output_padding=out_padding), nn.ReLU(True))

    def forward(self, input):
        
        x = self.trans(input)

        return x


#Mixup module
class MixUp(nn.Module):
    def __init__(self, m=-0.88):
        super(MixUp, self).__init__()
        w = nn.parameter.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2):
        mix_factor = self.sig(self.w)

        #expand mix factor to the size of x1 and x2 to perform element-wise multiplication
        x1_mix_factor = mix_factor.expand_as(x1)
        x2_mix_factor = mix_factor.expand_as(x2)
        
        out = x1 * x1_mix_factor + x2 *(1 -  x2_mix_factor)

        return out
    

class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()

        self.conv1 = nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()
    
    def forward(self, input):
        x = self.avg_pooling(input)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sig(x)

        return torch.mul(input, x)

class PixelAttention(nn.Module):
    def __init__(self, channel):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sig(x)

        return torch.mul(input, x)
    


class FABlock(nn.Module):
    def __init__(self, dim, kernel_size):
        super(FABlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(dim)
        self.pa = PixelAttention(dim)
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        y = input + x
        y = self.conv2(y)
        y = self.ca(y)
        y = self.pa(y)

        return torch.add(input, y)


# Deformable Convolution implementation from
# https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2
class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x

class DehazeNetwork(nn.Module):
    def __init__(self, inp, out, mc = 64):
        super(DehazeNetwork, self).__init__()
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   ConvMainBlock(inp, mc, kernel_size=7, padding=0),
                                   )
        self.down2 = ConvMainBlock(mc, mc*2, kernel_size=3, stride=2, padding=1)
        self.down3 = ConvMainBlock(mc*2, mc*4, kernel_size=3, stride=2, padding=1)

        self.mix1 = MixUp(m=4)
        self.mix2 = MixUp(m=1)
        self.fa_block = FABlock(mc*4, kernel_size=3)
        self.dcn_block = DeformableConv2d(256, 256)


        self.up1 = TransposeConv(mc*4, mc*2, kernel_size=3, stride=2, padding=1, out_padding=1)
        self.up2 = TransposeConv(mc*2, mc, kernel_size=3, stride=2, padding=1, out_padding=1)
                                
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 TransposeConv(mc, out, kernel_size=7, padding=0))
    
    def forward(self, input):
        #x_deconv = self.deconv(input) # preprocess
        
        x_down_1 = self.down1(input)
       
        x_down_2 = self.down2(x_down_1)
        
        x_down_3 = self.down3(x_down_2)

        x1 = self.fa_block(x_down_3)
        x2 = self.fa_block(x1)
        x3 = self.fa_block(x2)
        x4 = self.fa_block(x3)
        x5 = self.fa_block(x4)
        x6 = self.fa_block(x5)



       

        x_dcn_1 = self.dcn_block(x6)
        #print(x_dcn_1.shape)
        x_dcn_2 = self.dcn_block(x_dcn_1)
        #print(x_dcn_1.shape)

        x_mix = self.mix1(x_down_3, x_dcn_2)
        #print(x_mix.shape)
        x_up_1 = self.up1(x_mix)
        #print(x_up_1.shape)
        #print(x_down_2.shape)
        x_up_1_mix = self.mix2(x_down_2, x_up_1)
        x_up_2 = self.up2(x_up_1_mix)
        #print(x_up_2.shape)
        out = self.up3(x_up_2)
        #print(out.shape)

        return out

#pretrained vgg19
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

#contrast loss
class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss
    




