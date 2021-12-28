from monai.losses import DiceLoss
import numpy as np
import torch
import warnings
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss

def get_DiceLoss():

    cost_function = DiceLoss(include_background=True,softmax=True,to_onehot_y=True,squared_pred=True,reduction="none")
   
    cost_function.name = "Dice_Loss"
    return cost_function




class DiceCELoss(nn.Module):


    def __init__(self):
        super(DiceCELoss,self).__init__()

        self.dice_loss  = DiceLoss(include_background=True,softmax=True,to_onehot_y=True,squared_pred=True,reduction="none")
        self.ce_loss = CrossEntropyLoss(reduction='mean')
        self.name = "DiceCE_Loss"

    def forward(self,logits,target,weights):

        #Assign weight Tensor to CE class
        self.ce_loss.weight = weights
        
        dice_loss = self.dice_loss(logits,target)
        dice_loss = torch.mean(dice_loss,axis=0)
        #print(f"Dice Loss {dice_loss} | Weights = {weights}")
        dice_loss = dice_loss*weights
        dice_loss = dice_loss.sum()/weights.sum()

        

        ce_loss = self.ce_loss(logits,target.squeeze(1).long())

        #print(f"Dice Loss {dice_loss}")
        #print(f"CE Loss {ce_loss}")

        loss = dice_loss + ce_loss

        return loss




def median_frequency(y):
    unique, histo = np.unique(y, return_counts=True)

    if len(unique)==1:
        #warnings.warn("Median Frequency calculation expected more than 1 class, returning torch.tensor([1,1])")
        return None

    freq = histo / np.sum(histo)
    med_freq = np.median(freq)
    weights = np.asarray(med_freq / freq)
    return torch.from_numpy(weights).float()




def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def ignore_background(y_pred,y):

    #FROM MONAI
    """
    This function is used to remove background (the first channel) for `y_pred` and `y`.
    Args:
        y_pred: predictions. As for classification tasks,
            `y_pred` should has the shape [BN] where N is larger than 1. As for segmentation tasks,
            the shape should be [BNHW] or [BNHWD].
        y: ground truth, the first dim is batch.
    """
    y = y[:, 1:] if y.shape[1] > 1 else y
    y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred
    return y_pred, y

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None,include_background=True):

    ##https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/metrics.py
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """
    if not include_background:
        input,target = ignore_background(input,target)
    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"


    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

    