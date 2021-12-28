""" Parts of the U-Net model """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,se_block="PE"):
        super().__init__()
        self.se_block = se_block
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

        if se_block == "SE":
            self._se_block =  ChannelSELayer3D(out_channels)#Squeeze_Excite_Block(out_channels)
        elif se_block == "CSSE":
            self._se_block = ChannelSpatialSELayer3D(out_channels)
        elif se_block == "PE":
            self._se_block = ProjectExciteLayer(out_channels)

    def forward(self, x):

        x = self.double_conv(x)
        if self.se_block:
            x = self._se_block(x)
        return x

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,se_block="SE"):
        super(ResidualConv, self).__init__()

        self.se_block = se_block
        if not mid_channels:
            mid_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv3d(in_channels,out_channels, kernel_size=1,stride=1,bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
        )

        if se_block == "SE":
            self._se_block = ChannelSELayer3D(out_channels)#Squeeze_Excite_Block(out_channels)
        elif se_block == "CSSE":
            self._se_block = ChannelSpatialSELayer3D(out_channels)
        elif se_block == "PE":
            self._se_block = ProjectExciteLayer(out_channels)

    def forward(self, x):

        x = self.conv_block(x) + self.conv_skip(x)
        if self.se_block:
            x = self._se_block(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,residual_block=True,se_block="PE",pooling_size=(2,2,2)):
        super().__init__()
        if residual_block:
            self.conv = ResidualConv(in_channels,out_channels,se_block=se_block)
        else:
            self.conv = DoubleConv(in_channels, out_channels,se_block=se_block)

        self.pool = nn.MaxPool3d(pooling_size)




    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True,residual_block=True,se_block="PE",attention=True,MHCA_heads=0,multi_scale_sc="None",up_channels=0,up_scale_factor=(2,2,2)):
        super().__init__()

        self.attention = attention
        self.multi_scale_sc = multi_scale_sc
        if multi_scale_sc != "R":
            up_channels = in_channels #multiscale skip connection
        else:
            self.upsample = nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True)
        if attention:
            if MHCA_heads != 0:
                self.attention_block = CrossAttention(in_channels//2,in_channels)#MHCA(in_channels=in_channels,embedding_channels=in_channels,out_channels=in_channels//2,num_heads=MHCA_heads)
            #self.attention_block = Attention_block(F_g=in_channels//2,F_l=in_channels//2,F_int=in_channels//4) #what to do in trilinear case

        print(f"IN CHANNELS {in_channels} - UP CHANNELS {up_channels}")
        # if bilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=up_scale_factor, mode='trilinear', align_corners=True),
                                    nn.Conv3d(up_channels,in_channels//2,  kernel_size=3, padding=1))
            if residual_block:
                self.conv = ResidualConv(in_channels, out_channels, in_channels // 2,se_block=se_block)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,se_block=se_block)
        else:
            self.up = nn.ConvTranspose3d(up_channels , in_channels // 2, kernel_size=2, stride=2)
            if residual_block:
                self.conv = ResidualConv(in_channels, out_channels,se_block=se_block)
            else:
                self.conv = DoubleConv(in_channels, out_channels,se_block=se_block)


    def forward(self, x1, s):

        #print(f"UP X1 {x1.size()} , X2 {x2.size()}")

        #if self.attention:
        #    x = self.attention_block(x1,x2) #attention on skip connection signal

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        #if self.multi_scale_sc == "R":
        #    x_res = self.upsample(x1)
        #    x_res = equalize_dimensions(x_res,s)
        x1 = self.up(x1)
        x1 = equalize_dimensions(x1,s)
        x_in = torch.cat([s, x1], dim=1)
        #print(f"X up {x.size()}")
        x_out = self.conv(x_in)
        if self.multi_scale_sc == "R":
        #    #print(f"MultiScale X {x.size()} - x1 {x_res.size()}")
             x_out = torch.cat([x_out,x_in],dim=1)

        return x_out

def equalize_dimensions(x, upsampled_data):
    # print('x.shape = ', x.shape, ' != ', upsampled_data.shape, ' upsampled.shape')
    padding_dim1 = abs(x.shape[-1] - upsampled_data.shape[-1])
    padding_dim2 = abs(x.shape[-2] - upsampled_data.shape[-2])
    padding_dim3 = abs(x.shape[-3] - upsampled_data.shape[-3])

    if padding_dim1 % 2 == 0:
        padding_left = padding_dim1 // 2
        padding_right = padding_dim1 // 2
    else:
        padding_left = padding_dim1 // 2
        padding_right = padding_dim1 - padding_left

    if padding_dim2 % 2 == 0:
        padding_top = padding_dim2 // 2
        padding_bottom = padding_dim2 // 2
    else:
        padding_top = padding_dim2 // 2
        padding_bottom = padding_dim2 - padding_top

    if padding_dim3 % 2 == 0:
        padding_front = padding_dim3 // 2
        padding_back = padding_dim3 // 2
    else:
        padding_front = padding_dim3 // 2
        padding_back = padding_dim3 - padding_front

    pad_fn = nn.ConstantPad3d((padding_left, padding_right,
                            padding_top, padding_bottom,
                            padding_front, padding_back), 0)

    return pad_fn(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=4):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio

        
        #self.excite =nn.Sequential(nn.ReLU(inplace=True),
        #                          nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1),
        #                          nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1),
        #                          nn.Sigmoid(),)
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        #batch_size, num_channels, D, H, W = input_tensor.size()
        batch_size, num_channels, H, W, D = input_tensor.size()

        # Project:
        # Average along channels and different axes
        #squeeze_tensor_w =  nn.AdaptiveAvgPool3d((1, W, 1))(input_tensor)

        #squeeze_tensor_h =  nn.AdaptiveAvgPool3d((H, 1, 1))(input_tensor)

        #squeeze_tensor_d =  nn.AdaptiveAvgPool3d((1, 1, D))(input_tensor)

        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, W, 1))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (H, 1, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (1, 1, D))

        # tile tensors to original size and add:
        final_squeeze_tensor =  sum([squeeze_tensor_w.view(batch_size, num_channels, 1, W, 1),
                                    squeeze_tensor_h.view(batch_size, num_channels, H, 1, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, 1, 1, D)])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)
        #final_squeeze_tensor = self.excite(final_squeeze_tensor)
        #output_tensor = input_tensor * final_squeeze_tensor
        return output_tensor

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=4, num_channels=F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=2, padding=0, bias=True),
            nn.GroupNorm(num_groups=4, num_channels=F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        x1 = equalize_dimensions(x1,g1)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.upsample(psi)
        out = x * psi
        return out


class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class SA(nn.Module):

    def __init__(self,in_channels,embedding_channels,key_stride=1):
        super(SA, self).__init__()

        self.key_stride = key_stride
        factor = 1
        if self.key_stride == 2:
            factor = 2 #Sa used in MHCA therefore 
        self.kW = nn.Sequential(nn.Conv3d(in_channels//factor, embedding_channels, kernel_size=1,stride=1, bias=True),
                                nn.GroupNorm(num_groups=4, num_channels= embedding_channels))
        self.qW = nn.Sequential(nn.Conv3d(in_channels, embedding_channels, kernel_size=1, bias=True),
                                nn.GroupNorm(num_groups=4, num_channels= embedding_channels))
        self.vW = nn.Sequential(nn.Conv3d(in_channels//factor, embedding_channels, kernel_size=1, bias=True),
                                nn.GroupNorm(num_groups=4, num_channels= embedding_channels))

        self.psi = nn.Sequential(
            nn.Conv3d(embedding_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

        if self.key_stride == 2:
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self,g,x):
        #print(f"G {g.size()}")
        #print(f"X {x.size()}")
        key = self.kW(x)
        query = self.qW(g)
        value = self.vW(x)

        if self.key_stride ==2:
            key = nn.MaxPool3d(2)(key)

        #print(f"KEY {key.size()}")
        #print(f"QUERY {query.size()}")
        #print(f"VALUE {value.size()}")
        #query = equalize_dimensions(query,key)
        #key = equalize_dimensions(key,query)
        A = self.psi(self.relu(key + query))

        if self.key_stride == 2: #this indicates that SA was used in MHCA therefore the attention map needs to be upsampled
            A = self.upsample(A)
            A = equalize_dimensions(A,value)
        
        z = value * A
        return z


class Embedding(nn.Module):
    #https://github.com/HXLH50K/U-Net-Transformer/blob/main/models/utransformer/U_Transformer.py
    def __init__(self, channels):
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(channels, channels))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


    def forward(self, x):
        # x:[b, h*w*d, c]
        b, hwd, c = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))

        return x

class GlobalSpatialAttention(nn.Module):
    def __init__(self,in_channels):
        super(GlobalSpatialAttention,self).__init__()

        self.query = nn.Conv3d(in_channels,in_channels//8,kernel_size=3,padding=1)
        self.key = nn.Conv3d(in_channels,in_channels//8,kernel_size=3,padding=1)
        self.value = nn.Conv3d(in_channels,in_channels,kernel_size=3,padding=1)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):

        M = self.query(x)
        N = self.key(x)
        W = self.value(x)

        b,c,h,w,d = M.size()


        M = M.reshape(b,c,h*w*d)
        N = N.reshape(b,c,h*w*d)
        W = W.reshape(b,c*8,h*w*d) # W has no compression in channels

        
        B = self.softmax(torch.bmm(M.permute(0,2,1),N))
        #print(B.size())
        #W = b x c x (hwd) #B = b x (hwd) x (hwd)
        GSA = torch.bmm(W,B.permute(0,2,1)).reshape(b,c*8,h,w,d)


        return GSA

class SelfAwareAttention(nn.Module):
    def __init__(self,in_channels,MHTSA_heads=1,MHGSA_heads=1):
        super(SelfAwareAttention,self).__init__()

        self.MHTSA_heads = MHTSA_heads
        self.MHGSA_heads = MHGSA_heads
        if MHTSA_heads == 1:
            self.TSA = SelfAttention(in_channels)
        else:
            self.TSA = MHTSA(in_channels,MHTSA_heads)
        if MHGSA_heads == 1:
            self.GSA = GlobalSpatialAttention(in_channels)
        else:
            self.GSA = MHGSA(in_channels,MHTSA_heads)

        self.lambda_1 = nn.Parameter(torch.Tensor([0]))
        self.lambda_2 = nn.Parameter(torch.Tensor([0]))
        #self.lambda_3 = nn.Parameter(torch.Tensor([1/3]))
        

    def forward(self,x):

        TSA = self.TSA(x)
        GSA = self.GSA(x)

        return self.lambda_1*TSA + self.lambda_2*GSA + x

class SelfAttention(nn.Module):
    def __init__(self,in_channels):
        super(SelfAttention,self).__init__()
        self.position_encoding= PositionalEncodingPermute3D(in_channels)
        self.query = Embedding(in_channels)
        self.key = Embedding(in_channels)
        self.value = Embedding(in_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        b,c,h,w,d = x.size()
        pe = self.position_encoding(x)
        x += pe
        #Reshape Feature Map b * c * h * w * d -> b * (hwd) * c
        x = x.reshape(b, c, h * w * d).permute(0, 2, 1) # [b,hwd,c]
        

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Q = b * hwd * c , K^T = b * c * hwd
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(c))

        # A = b * hwd * hwd , V = b * hwd * c -> b * hwd * c -> b * c * hwd -> b * c * h * w * d                 
        x = torch.bmm(A,V).permute(0,2,1).reshape(b,c,h,w,d)

        return x

class CrossAttention(nn.Module):
    def __init__(self,in_channels_s,in_channels_y):
        super(CrossAttention,self).__init__()
        self.position_encoding_s= PositionalEncodingPermute3D(in_channels_s)
        self.position_encoding_y= PositionalEncodingPermute3D(in_channels_y)

        #Prepare Skip connection for value embedding
        self.Sconv = nn.Sequential(nn.MaxPool3d(2),
                                nn.Conv3d(in_channels_s,in_channels_s,1),
                                nn.GroupNorm(num_groups=4, num_channels=in_channels_s),
                                nn.ReLU(inplace=True))

        #Prepare high level feature map Y for Key and Query embedding
        self.Yconv = nn.Sequential(nn.Conv3d(in_channels_y,in_channels_s,1),
                                nn.GroupNorm(num_groups=4, num_channels=in_channels_s),
                                nn.ReLU(inplace=True))

        self.Yupsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                                    nn.Conv3d(in_channels_y,in_channels_s,3,padding=1),
                                    nn.Conv3d(in_channels_s,in_channels_s,1),
                                    nn.GroupNorm(num_groups=4, num_channels=in_channels_s),
                                    nn.ReLU(inplace=True))

        self.Zupsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                                    nn.Conv3d(in_channels_s,in_channels_s,3,padding=1))

        self.query = Embedding(in_channels_s)
        self.key = Embedding(in_channels_s)
        self.value = Embedding(in_channels_s)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,y,s):

        pe_s = self.position_encoding_s(s)
        pe_y = self.position_encoding_y(y)
        s = s + pe_s
        y = y + pe_y

        s1 = self.Sconv(s)
        y1 = self.Yconv(y)

        

        #print(f"S {s.size()}")
        #print(f"Y {y.size()}")

        #Sb,Sc,Sh,Sw,Sd = s.size()
        #Yb,Yc,Yh,Yw,Yd = y.size() # S dims and Y dims should be equivalent
        b,c,h,w,d = y1.size()
        #Reshape Feature Map b * c * h * w * d -> b * (hwd) * c
        s1 = s1.reshape(b, c, h * w * d).permute(0, 2, 1) # [b,hwd,c]
        y1 = y1.reshape(b, c, h * w * d).permute(0, 2, 1) # [b,hwd,c]
        

        Q = self.query(y1)
        K = self.key(y1)
        V = self.value(s1)

        # Q = b * hwd * c , K^T = b * c * hwd
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(c)) 

        # A = b * hwd * hwd , V = b * hwd * c -> b * hwd * c -> b * c * hwd -> b * c * h * w * d                 
        z = torch.bmm(A,V).permute(0,2,1).reshape(b,c,h,w,d)

        z = self.Zupsample(z)

        z = equalize_dimensions(z,s)
        y = self.Yupsample(y)
        y = equalize_dimensions(y,s)
        s = z * s
        
        #print(f"Z {z.size()}")
        #print(f"Y_up {y_up.size()}")
        #y_up = equalize_dimensions(y_up,z)
        x = torch.cat([s,y],dim=1)

        return x


class PositionalEncoding3D(nn.Module):
    #https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels/6)*2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z

        return emb[None,:,:,:,:orig_ch].repeat(batch_size, 1, 1, 1, 1)

class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)        
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0,2,3,4,1)
        enc = self.penc(tensor)
        return enc.permute(0,4,1,2,3)   
class MHTSA(nn.Module):

    def __init__(self,in_channels,num_heads):
        super(MHTSA, self).__init__()


        self.heads = nn.ModuleList([SelfAttention(in_channels) for i in range(num_heads)])
        self.conv =  nn.Sequential(nn.Conv3d(in_channels*num_heads, in_channels, kernel_size=3,padding=1, bias=True),
                                nn.GroupNorm(num_groups=4, num_channels= in_channels))

    def forward(self,x):
        z_heads = []
        for head in self.heads:
            z_heads.append(head(x))
        
        z = torch.cat(z_heads,dim=1)
        z = self.conv(z)

        return z

class MHGSA(nn.Module):

    def __init__(self,in_channels,num_heads):
        super(MHGSA, self).__init__()


        self.heads = nn.ModuleList([GlobalSpatialAttention(in_channels) for i in range(num_heads)])
        self.conv =  nn.Sequential(nn.Conv3d(in_channels*num_heads, in_channels, kernel_size=3,padding=1, bias=True),
                                nn.GroupNorm(num_groups=4, num_channels= in_channels))

    def forward(self,x):
        z_heads = []
        for head in self.heads:
            z_heads.append(head(x))
        
        z = torch.cat(z_heads,dim=1)
        z = self.conv(z)

        return z

class MHCA(nn.Module):

    def __init__(self,in_channels,embedding_channels,out_channels,num_heads):
        super(MHCA, self).__init__()


        self.heads = nn.ModuleList([SA(in_channels=in_channels,embedding_channels=embedding_channels,key_stride=2) for i in range(num_heads)])
        self.conv =  nn.Sequential(nn.Conv3d(embedding_channels*num_heads, out_channels, kernel_size=1, bias=True),
                                nn.GroupNorm(num_groups=4, num_channels= out_channels))

    def forward(self,g,x):
        z_heads = []
        for head in self.heads:
            z_heads.append(head(g,x))
        
        z = torch.cat(z_heads,dim=1)
        z = self.conv(z)

        return z