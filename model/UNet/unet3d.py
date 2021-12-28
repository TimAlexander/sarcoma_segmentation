
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

from .unet_parts3d import *


class ResUNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, f_maps=16,levels=4,residual_block=True,se_block="CSSE",attention=True,trilinear=True,MHTSA_heads=0,MHGSA_heads=0,MSSC="None"):
        super(ResUNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear
        self.MSSC = MSSC
        feature_maps = (f_maps * np.exp2(np.arange(levels+1))).astype(np.int)
   

        #For multiscale skip connections
        up_channels = None
        if self.MSSC == 'D':
            up_channels = np.cumsum(feature_maps[::-1])[::-1]
        elif self.MSSC == 'R':
            if levels == 4:
                up_channels = [128,192,96,48,24][::-1]
            else:
                raise NotImplementedError("Residual MC just implemented for 4 levels")

        else:
            up_channels = feature_maps[::-1]

        #print(f"FMAPS {f_maps} - feature_maps {feature_maps} - up_channels {up_channels}")

        self.inc = DoubleConv(n_channels, f_maps,se_block=se_block)
        pooling_sizes = [(2,2,2),(2,2,2),(2,2,2),(2,2,2)]
        #self.encoder = nn.ModuleList([Down(f_maps*(2**i),f_maps*(2**(i+1)),residual_block=residual_block,se_block=se_block) for i in range(levels-1)]) #last level is for bridge
        self.encoder = nn.ModuleList([Down(feature_maps[i],feature_maps[i+1],residual_block=residual_block,se_block=se_block,pooling_size=pooling_sizes[i]) for i in range(levels-1)])


        self.bridge = nn.Sequential(Down(feature_maps[-2],feature_maps[-1],residual_block=residual_block,se_block=se_block),
                                        MHTSA(feature_maps[-1],MHTSA_heads))
                                        #SelfAwareAttention(feature_maps[-1],MHTSA_heads,MHGSA_heads))
                                     

        #self.decoder  =    ([Up(f_maps*(2**(i+1)),f_maps*(2**i) ,trilinear,residual_block=residual_block,se_block=se_block,attention=attention,MHCA_heads=MHCA_heads) for i in range(1,levels)[::-1]])
        #self.decoder.append(Up(f_maps*2,f_maps,trilinear,residual_block=residual_block,se_block=se_block,attention=attention,MHCA_heads=MHCA_heads))
        up_scale_factors = [("dummy")] + pooling_sizes
        self.decoder = ([Up(feature_maps[i],feature_maps[i-1],trilinear,residual_block=residual_block,se_block=se_block,attention=attention,MHCA_heads=0,multi_scale_sc=MSSC,up_channels=up_channels[i],up_scale_factor=up_scale_factors[i]) for i in range(1,levels+1)[::-1]])
        self.decoder = ModuleList(self.decoder)

        
        
        if self.MSSC == 'R':
            self.outc = OutConv(up_channels[0], n_classes)
        else:
            self.outc = OutConv(f_maps, n_classes)

    def forward(self, x):
        encoding_features = []
        x = self.inc(x)
        encoding_features.append(x)
        for block in self.encoder:
            x= block(x)
            #print(f"Encoding {x.size()}")
            encoding_features.append(x)

        x = self.bridge(x)

        #print(f"Bridge {x.size()}")
        for block,feature in zip(self.decoder,encoding_features[::-1]):
            x = block(x,feature)
            #print(f"Decoding {x.size()}")

        logits = self.outc(x)
        return logits