import torch.nn as nn
import torch
import torch.nn.functional as F
from monai.networks.blocks import SimpleASPP
import math

class BoBNet(nn.Module):

    def __init__(self,num_classes=2, init_weights=True, channels=1,dropout=0.5):
        super(BoBNet, self).__init__()

        self.channels = channels
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv16 = nn.Sequential(
            nn.Conv2d(self.channels, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv32 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv64 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv128 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pyramid = SPPLayer(levels=[1, 2, 4], pool_type='max_pool')

        self.linear = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(21 * 128, 128),  # TODO: Calculate the 21 from the levels (4*4 + 2*2 + 1*1)
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            # TODO: One more dropout?
            nn.Linear(128, self.num_classes)
        )


        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv16(x)
        x = self.conv32(x)
        x = self.conv64(x)
        x = self.conv128(x)
        x = self.pyramid(x)
        x = self.linear(x)
        return x


class SPPLayer(torch.nn.Module):

    # 定义Layer需要的额外参数（除Tensor以外的）
    def __init__(self, levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.levels = levels
        self.pool_type = pool_type

    # forward()的参数只能是Tensor(>=0.4.0) Variable(< 0.4.0)
    def forward(self, x):
        # num:样本数量 c:通道数 h:高 w:宽
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        num, c, h, w = x.size()
        for i in range(len(self.levels)):

            '''
            The equation is explained on the following site:
            http://www.cnblogs.com/marsggbo/p/8572846.html#autoid-0-0-0
            '''
            kernel_size = (math.ceil(h / self.levels[i]), math.ceil(w / self.levels[i]))  # kernel_size = (h, w)
            padding = (
                math.floor((kernel_size[0] * self.levels[i] - h + 1) / 2),
                math.floor((kernel_size[1] * self.levels[i] - w + 1) / 2))

            # update input data with padding
            #  class torch.nn.ZeroPad2d(padding)[source]
            #
            #     Pads the input tensor boundaries with zero.
            #
            #     For N`d-padding, use :func:`torch.nn.functional.pad().
            #     Parameters:	padding (int, tuple) – the size of the padding. If is int, uses the same padding in all boundaries.
            # If a 4-tuple, uses (paddingLeft, paddingRight, paddingTop, paddingBottom)
            zero_pad = torch.nn.ZeroPad2d((padding[1], padding[1], padding[0], padding[0]))
            x_new = zero_pad(x)

            # update kernel and stride
            h_new, w_new = x_new.size()[2:]

            kernel_size = (math.ceil(h_new / self.levels[i]), math.ceil(w_new / self.levels[i]))
            stride = (math.floor(h_new / self.levels[i]), math.floor(w_new / self.levels[i]))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
            elif self.pool_type == 'avg_pool':
                tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
            else:
                raise RuntimeError("Unknown pooling type")

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)

        return x_flatten

#https://github.com/revidee/pytorch-pyramid-pooling/blob/master/pyramidpooling.py
class PyramidPooling(nn.Module):
    def __init__(self, levels, mode="max"):
        """
        General Pyramid Pooling class which uses Spatial Pyramid Pooling by default and holds the static methods for both spatial and temporal pooling.
        :param levels defines the different divisions to be made in the width and (spatial) height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n], where  n: sum(filter_amount*level*level) for each level in levels (spatial) or
                                                                    n: sum(filter_amount*level) for each level in levels (temporal)
                                            which is the concentration of multi-level pooling
        """
        super(PyramidPooling, self).__init__()
        self.levels = levels
        self.mode = mode

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out

    @staticmethod
    def spatial_pyramid_pool(previous_conv, levels, mode):
        """
        Static Spatial Pyramid Pooling method, which divides the input Tensor vertically and horizontally
        (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width and height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(levels)):
            h_kernel = int(math.ceil(previous_conv_size[0] / levels[i]))
            w_kernel = int(math.ceil(previous_conv_size[1] / levels[i]))
            w_pad1 = int(math.floor((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            h_pad2 = int(math.ceil((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * levels[i] - previous_conv_size[1]) and \
                   h_pad1 + h_pad2 == (h_kernel * levels[i] - previous_conv_size[0])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)

        return spp

    @staticmethod
    def temporal_pyramid_pool(previous_conv, out_pool_size, mode):
        """
        Static Temporal Pyramid Pooling method, which divides the input Tensor horizontally (last dimensions)
        according to each level in the given levels and pools its value according to the given mode.
        In other words: It divides the Input Tensor in "level" horizontal stripes with width of roughly (previous_conv.size(3) / level)
        and the original height and pools the values inside this stripe
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)
            #
            h_kernel = previous_conv_size[0]
            w_kernel = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            w_pad1 = int(math.floor((w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * out_pool_size[i] - previous_conv_size[1])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                tpp = x.view(num_sample, -1)
            else:
                tpp = torch.cat((tpp, x.view(num_sample, -1)), 1)

        return tpp


class SpatialPyramidPooling(PyramidPooling):
    def __init__(self, levels, mode="max"):
        """
                Spatial Pyramid Pooling Module, which divides the input Tensor horizontally and horizontally
                (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
                Can be used as every other pytorch Module and has no learnable parameters since it's a static pooling.
                In other words: It divides the Input Tensor in level*level rectangles width of roughly (previous_conv.size(3) / level)
                and height of roughly (previous_conv.size(2) / level) and pools its value. (pads input to fit)
                :param levels defines the different divisions to be made in the width dimension
                :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
                :returns (forward) a tensor vector with shape [batch x 1 x n],
                                                    where n: sum(filter_amount*level*level) for each level in levels
                                                    which is the concentration of multi-level pooling
                """
        super(SpatialPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        """
                Calculates the output shape given a filter_amount: sum(filter_amount*level*level) for each level in levels
                Can be used to x.view(-1, spp.get_output_size(filter_amount)) for the fully-connected layers
                :param filters: the amount of filter of output fed into the spatial pyramid pooling
                :return: sum(filter_amount*level*level)
        """
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out


class TemporalPyramidPooling(PyramidPooling):
    def __init__(self, levels, mode="max"):
        """
        Temporal Pyramid Pooling Module, which divides the input Tensor horizontally (last dimensions)
        according to each level in the given levels and pools its value according to the given mode.
        Can be used as every other pytorch Module and has no learnable parameters since it's a static pooling.
        In other words: It divides the Input Tensor in "level" horizontal stripes with width of roughly (previous_conv.size(3) / level)
        and the original height and pools the values inside this stripe
        :param levels defines the different divisions to be made in the width dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns (forward) a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        super(TemporalPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x):
        return self.temporal_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        """
        Calculates the output shape given a filter_amount: sum(filter_amount*level) for each level in levels
        Can be used to x.view(-1, tpp.get_output_size(filter_amount)) for the fully-connected layers
        :param filters: the amount of filter of output fed into the temporal pyramid pooling
        :return: sum(filter_amount*level)
        """
        out = 0
        for level in self.levels:
            out += filters * level
        return out
