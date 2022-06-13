import torch
print(torch.__version__)
from torch import nn as nn

"""
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride)
Every conv is a same convulation.
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
# Tuple: (out_channels, kernel_size, stride) # every convoluational in yolo V3 uses the same pattern
#List: ["B",1], block, number of repeats 1
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],     # to this point is darknet-53. Uses 53 CNNs. Darknet-53 is a convolutional neural network that acts as a backbone for the YOLOv3 object detection approach. The improvements upon its predecessor Darknet-19 include the use of residual connections, as well as more layers.
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",            #The Upsampling layer is a simple layer with no weights that will double the dimensions of input and can be used in a generative model when followed by a traditional convolutional layer.
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act = True, **kwargs): # **kwargs means keyword arguments. Its actually not a python keyword, its just a convention, that people follow. That will get all the key-value parameters passed to the function.
        super().__init__() #super() builtin returns a proxy object (temporary object of the superclass) that allows us to access methods of the base class
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs) # kwargs are kernel_size, stride. #  conv2D is the function to perform convolution to a 2D data (e.g, an image)
        self.bn = nn.BatchNorm2d(out_channels) # BatchNorm2d is the number of dimensions/channels that output from the last layer and come in to the batch norm layer.
        self.leaky = nn.LeakyReLU(0.1) #activation function, LeakyReLU function is used to fix a part of the parameters to cope with the gradient death. 
        self.use_bn_act = bn_act #batch normalization layer

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ResidualBlock(nn.Module): #FOR B
    def __init__(self, channels, use_residual = True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                CNNBlock(channels, channels//2, kernel_size=1),
                CNNBlock(channels//2, channels, kernel_size=3, padding=1))
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x




class ScalePrediction(nn.Module): # FOR S
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1), # Padding is a term relevant to convolutional neural networks as it refers to the amount of pixels added to an image when it is being processed by the kernel of a CNN. For example, if the padding in a CNN is set to zero, then every pixel value that is added will be of value zero.
            CNNBlock(2*in_channels, 3 * (num_classes + 5), bn_act=False, kernel_size=1), #3 anchors for each cell, for every anchor box we need one node for each of the classes we want to predict
        ) #[po, w, y, w, h] = 5  probabilities that each bounding box has, po=probab that there is an object in that cell
        self.num_classes = num_classes


    def forward(self, x):
        return(
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]) # making into two different dimensions
            .permute(0, 1, 3, 4, 2)
        )
        # N X 3 X 13 X 13 X 5+NUMCLASSES
        # N X 3 X 26 X 26 X 5+NUMCLASSES

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs




    def _create_conv_layers(self): # to track all the layers
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels
            
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
            
        return layers

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
