import torch
from torch import nn

from darknet53 import Darknet53
from darknet53 import DarkResidualBlock
from darknet53 import dbl

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
    def __init__(self):
        super(YOLOv3, self).__init__()
        self.darknet = Darknet53(DarkResidualBlock)
        self.conv_block1 = dbl_block(1024, 512)
        self.final_block1 = final_block(512, 255)

        self.conv_block2 = dbl_block(768, 256)
        self.final_block2 = final_block(256, 255)

        self.upsample1 = up_sample(512, 256, 2)
        self.upsample2 = up_sample(256, 128, 2)

        self.conv_block3 = dbl_block(384, 128)
        self.final_block3 = final_block(128, 255)

    def forward(self, x):
        f1, f2, f3 = self.darknet(x) #[1024,13,13], [512, 26,26], [256,52,52]

        '''feature Map 1 추출'''

        x1 = self.conv_block1(f1)
        print("x1 shape:", x1.shape) #[1, 512, 13, 13]

        featureMap1=self.final_block1(x1)
        print("**feature Map1** shape:", featureMap1.shape) #[1, 255, 13, 13] => 255인 이유

        '''feature Map 2 추출'''

        x2 = self.upsample1(x1)
        print(x2.shape) #[1, 256, 26, 26]
        x2 = torch.cat((x2, f2), dim=1)
        print(x2.shape) #[1, 768, 26, 26]

        x2 = self.conv_block2(x2)
        print("x2 shape:",  x2.shape) #[1, 256, 26, 26]

        featureMap2 = self.final_block2(x2)
        print("**feature Map2** shape:",featureMap2.shape)  # [1, 255, 26, 26] => 255인 이유

        '''feature Map 3 추출'''
        x3 = self.upsample2(x2)
        print(x3.shape)  # [1, 128, 52, 52]
        x3 = torch.cat((x3, f3), dim=1)
        print(x3.shape)  # [1, 384, 52, 52]

        x3 = self.conv_block3(x3)
        print("x3 shape:", x3.shape)  # [1, 256, 26, 26]

        featureMap3 = self.final_block3(x3)
        print("**feature Map3** shape:",featureMap3.shape)  # [1, 255, 26, 26] => 255인 이유

        return featureMap1, featureMap2, featureMap3

if __name__ == '__main__':
    input_image = torch.randn(1, 3, 416, 416, dtype=torch.float)
    model = YOLOv3()
    f1,f2,f3 = model.forward(input_image)