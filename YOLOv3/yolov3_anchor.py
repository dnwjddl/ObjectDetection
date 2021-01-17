import torch
from torch import nn
import numpy as np

from darknet53 import Darknet53
from darknet53 import DarkResidualBlock
from darknet53 import dbl

from utils import *

from anchor import Detection

def dbl_block(in_channels, out_channels):
    double_channels = out_channels * 2

    return nn.Sequential(
        dbl(in_channels, out_channels, kernel_size=1, padding=0),
        dbl(out_channels, double_channels, kernel_size=3),
        dbl(double_channels, out_channels, kernel_size=1, padding=0),
        dbl(out_channels, double_channels, kernel_size=3),
        dbl(double_channels, out_channels, kernel_size=1, padding=0)
    )

def final_block(in_channels, out_channels):
    return nn.Sequential(
        dbl(in_channels, in_channels*2, kernel_size = 3),
        nn.Conv2d(in_channels*2, out_channels, kernel_size=1, stride = 1, padding = 0, bias = True)
    )

def up_sample(in_channels, out_channels, scale_factor):
    return nn.Sequential(
        dbl(in_channels, out_channels, kernel_size=1, padding=0),
        nn.Upsample(scale_factor= scale_factor, mode = 'nearest')
    )

class YOLOv3(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        anchors = {'scale1': [(10, 13), (16, 30), (33, 23)],
                   'scale2': [(30, 61), (62, 45), (59, 119)],
                   'scale3': [(116, 90), (156, 198), (373, 326)]}

        self.darknet = Darknet53(DarkResidualBlock)
        self.conv_block1 = dbl_block(1024, 512)
        self.final_block1 = final_block(512, 255)
        self.anchor1 = Detection(anchors['scale1'], 416, 80)

        self.conv_block2 = dbl_block(768, 256)
        self.final_block2 = final_block(256, 255)
        self.anchor2 = Detection(anchors['scale2'], 416, 80)

        self.upsample1 = up_sample(512, 256, 2)
        self.upsample2 = up_sample(256, 128, 2)

        self.conv_block3 = dbl_block(384, 128)
        self.final_block3 = final_block(128, 255)
        self.anchor3 = Detection(anchors['scale3'], 416, 80)

    def forward(self, x):
        f1, f2, f3 = self.darknet(x) #[1024,13,13], [512, 26,26], [256,52,52]

        '''feature Map 1 추출'''

        x1 = self.conv_block1(f1)
        print("x1 shape:", x1.shape) #[1, 512, 13, 13]

        featureMap1=self.final_block1(x1)
        print("**feature Map1** shape:", featureMap1.shape) #[1, 255, 13, 13] => 255인 이유

        print("===========Detecting output1============")
        output1 = self.anchor1(featureMap1)
        print("===========Detected output1============")

        '''feature Map 2 추출'''

        x2 = self.upsample1(x1)
        print(x2.shape) #[1, 256, 26, 26]
        x2 = torch.cat((x2, f2), dim=1)
        print(x2.shape) #[1, 768, 26, 26]

        x2 = self.conv_block2(x2)
        print("x2 shape:",  x2.shape) #[1, 256, 26, 26]

        featureMap2 = self.final_block2(x2)
        print("**feature Map2** shape:",featureMap2.shape)  # [1, 255, 26, 26] => 255인 이유

        print("===========Detecting output2============")
        output2 = self.anchor2(featureMap2)
        print("===========Detected output2============")

        '''feature Map 3 추출'''
        x3 = self.upsample2(x2)
        print(x3.shape)  # [1, 128, 52, 52]
        x3 = torch.cat((x3, f3), dim=1)
        print(x3.shape)  # [1, 384, 52, 52]

        x3 = self.conv_block3(x3)
        print("x3 shape:", x3.shape)  # [1, 256, 26, 26]

        featureMap3 = self.final_block3(x3)
        print("**feature Map3** shape:",featureMap3.shape)  # [1, 255, 26, 26] => 255인 이유

        print("===========Detecting output3============")
        output3 = self.anchor3(featureMap3)
        print("===========Detected output3============")


        yolo_outputs = [output1, output2, output3]
        yolo_outputs = torch.cat(yolo_outputs, 1 ).detach()

        result = results(yolo_outputs, 0.5)

        return yolo_outputs, result

    def load_weight(self, weightfile):
        #Open the weightfile
        fp = open(weightfile, "rb")
        weights = np.fromfile(fp, dtype = np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

        # If module_type is convolutional load weights
        # Otherwise ignore.

        if module_type == "convolutional":
            model = self.module_list[i]
            try:
                batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
            except:
                batch_normalize = 0

            conv = model[0]

        if (batch_normalize):
            bn = model[1]

            # Get the number of weights of Batch Norm Layer
            num_bn_biases = bn.bias.numel()

            # Load the weights
            bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
            ptr += num_bn_biases

            bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
            ptr += num_bn_biases

            bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
            ptr += num_bn_biases

            bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
            ptr += num_bn_biases

            # Cast the loaded weights into dims of model weights.
            bn_biases = bn_biases.view_as(bn.bias.data)
            bn_weights = bn_weights.view_as(bn.weight.data)
            bn_running_mean = bn_running_mean.view_as(bn.running_mean)
            bn_running_var = bn_running_var.view_as(bn.running_var)

            # Copy the data to model
            bn.bias.data.copy_(bn_biases)
            bn.weight.data.copy_(bn_weights)
            bn.running_mean.copy_(bn_running_mean)
            bn.running_var.copy_(bn_running_var)
        else:
            # Number of biases
            num_biases = conv.bias.numel()

            # Load the weights
            conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
            ptr = ptr + num_biases

            # reshape the loaded weights according to the dims of the model weights
            conv_biases = conv_biases.view_as(conv.bias.data)

            # Finally copy the data
            conv.bias.data.copy_(conv_biases)

        #Let us load the weights for the Convolutional Layers
        num_weights = conv.weight.numel()

        # Do the same as above for weights
        conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
        ptr = ptr + num_weights

        conv_weights = conv_weights.view_as(conv.weight.data)
        conv.weight.data.copy_(conv_weights)

if __name__ == '__main__':
    input_image = torch.randn(1, 3, 416, 416, dtype=torch.float)
    model = YOLOv3(80)
    #model.load_weights("yolov3.weights")
    output1, output2 = model.forward(input_image)
    print(output1.shape) #torch.Size([1, 10647, 85])
    print(output2.shape) #torch.Size([random, 8])

    '''
    # 8개의 속성
    ## batch 에서 이미지 인덱스
    ## 2~5 꼭지점 좌표
    ## 6 objectness 점수
    ## 7 maximum confidence 를 가진 class 점수
    ## 8 그 class의 index
    '''