
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

from .unet_parts2d import *


class ResUNet2D(nn.Module):
    def __init__(self, n_channels, n_classes, f_maps=16,levels=4,residual_block=True,se_block=True,attention=True,bilinear=True):
        super(ResUNet2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, f_maps,se_block=se_block)

        self.encoder = nn.ModuleList([Down(f_maps*(2**i),f_maps*(2**(i+1)),residual_block=residual_block,se_block=se_block) for i in range(levels-1)]) #last level is for bridge
        
        self.bridge = Down(f_maps*(2**(levels-1)),f_maps*(2**(levels)) //factor,residual_block=residual_block,se_block=False)
        
        self.decoder = [Up(f_maps*(2**(i+1)),f_maps*(2**i) // factor,bilinear,residual_block=residual_block,attention=attention) for i in range(1,levels)[::-1]]
        self.decoder.append(Up(f_maps*2,f_maps,bilinear,residual_block=residual_block,attention=attention))
        self.decoder = ModuleList(self.decoder)
        
        self.outc = OutConv(f_maps, n_classes)

    def forward(self, x):
        encoding_features = []
        x = self.inc(x)
        encoding_features.append(x)
        for block in self.encoder:
            x= block(x)
            encoding_features.append(x)

        x = self.bridge(x)

        for block,feature in zip(self.decoder,encoding_features[::-1]):
            x = block(x,feature)

        logits = self.outc(x)
        return logits