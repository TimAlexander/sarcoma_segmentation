import monai.networks.nets as mnn
import  torch.nn as nn
import torch
import numpy as np
from .UNet.model import UNet3D, ResidualUNet3D
from .UNet.unet2d import ResUNet2D
from .UNet.unet3d import ResUNet3D
from .CBRTinny import CBRTinny
from .BoBNet import BoBNet
from .Quicknat import QuickNat


DenseNets = ['densenet121','densenet161','densenet169','densenet201']
ResNets = ['resnet18','resnet34','resnet50','resnet101','resnet152']

def load_model(model_name,hparams):


    if model_name == "3DUNet":

        #model = mnn.UNet(dimensions=3,in_channels=1,out_channels=2,
        #                channels=(16, 32, 64, 128, 256),
        #                strides=(2, 2, 2, 2),
        #                num_res_units=2)
        model = UNet3D(in_channels=1,out_channels=2,f_maps=16
        ,final_sigmoid=False,is_segmentation=False,num_levels=5)
        model.name = model_name

        return model
    if model_name == "3DResUNet":

        model = ResUNet3D(n_channels=1, n_classes=2, 
        f_maps=hparams["f_maps"], 
        levels=hparams["levels"],
        residual_block=hparams["residual_block"],
        se_block = hparams["se_block"],
        attention=hparams["attention"],
        trilinear=hparams["trilinear"],
        MHTSA_heads=hparams["MHTSA_heads"],
        MHGSA_heads=hparams["MHGSA_heads"],
        MSSC=hparams["MSSC"])
        model.name = model_name
        
        return model

    if model_name == "2DUNet":

        model = UNet2D(n_channels=1,n_classes=2,bilinear=False)
        model.name = model_name

        return model
    
    if model_name == "2DResUNet":

        model = ResUNet2D(n_channels=1, n_classes=2, 
        f_maps=hparams["f_maps"], 
        levels=hparams["levels"],
        residual_block=hparams["residual_block"],
        se_block = hparams["se_block"],
        attention=hparams["attention"],
        bilinear=hparams["bilinear"])
        model.name = model_name
        model.name = model_name

        return model

    if model_name == "QuickNat":
        model = QuickNat({'num_channels':1,
                        'num_filters':64,
                        'kernel_h':5,
                        'kernel_w':5,
                        'kernel_c':1,
                        'stride_conv':1,
                        'pool':2,
                        'stride_pool':2,
                        'num_class':2,
                        'se_block': 'NONE',
                        'drop_out':0.2})
        model.name = model_name
        return model

    if model_name == "TransUNet":

        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = 2
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(224 /16), int(224 / 16))
        model  = ViT_seg(config_vit, img_size=224, num_classes=config_vit.n_classes)
        model.load_from(weights=np.load(config_vit.pretrained_path))

        model.name = model_name
        return model
    
    if model_name == "VNet":
        
        model = mnn.VNet(out_channels=2,dropout_prob=0,act='prelu')
        
        model.name = model_name
        
        return model


    if model_name == "cbrtiny":
        model = CBRTinny(num_classes=2,channels=1,drop_rate=0)
        model.name = model_name

        return model

    if model_name == "bobnet":
        model = BoBNet(channels=1,dropout=hparams["dropout"])
        model.name = model_name

        return model

    if model_name in ResNets:

        model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=False)
        model.name = model_name
        #for param in list(model.layer4.children())[-1].parameters():
        #    param.requires_grad = True
                                    
                  
        return change_classifier(model)
    ##DenseNets
    if model_name in DenseNets:
        
        model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True,drop_rate=dropout)
        model.name = model_name
        set_parameter_requires_grad(model,True)
        for param in model.features.denseblock4.denselayer16.parameters(): #16 for densenet121 32 for 201
            param.requires_grad = True
        #for param in model.features.denseblock4.denselayer31.parameters(): #16 for densenet121 32 for 201
        #    param.requires_grad = True

        return change_classifier(model)

    else:

        raise NameError("Model not found")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def change_classifier(model):
    classifier_name, old_classifier = model._modules.popitem()
    new_classifier = nn.Linear(old_classifier.in_features,2)
    model.add_module(classifier_name, new_classifier)
    
    return model
