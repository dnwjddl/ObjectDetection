import torch
from torch import nn

from darknet53 import Darknet53
from darknet53 import DarkResidualBlock
from darknet53 import dbl


class YOLOv3(nn.Module):
    def __init__(self):
        super(YOLOv3, self).__init__()
        self.darknet = Darknet53(DarkResidualBlock)

        self.dbl1 = dbl(512, 1024, kernel_size=3)
        self.conv1 = nn.Conv2d(1024,512, kernel_size = 1, stride = 1, padding =0, bias =True)

        self.dbl2 = dbl(1024, 512)
        self.dbl3 = dbl(512, 512)
        self.dbl4 = dbl(512, 256)
        self.dbl5 = dbl(1024, 256)

        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size = 3, padding = 1, stride = 1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.conv_block1 = conv_block(1024, 512)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        f1, f2, f3 = self.darknet(x) #[1024,13,13], [512, 26,26], [256,52,52]

        '''feature Map 1 추출'''

        x1 = self.(f1)
        print("x1 shape:", x1.shape) #[1, 512, 13, 13]

        x1_1 = self.dbl1(x1) ##이거
        print("x1_1 shape:", x1_1.shape)
        featureMap1 = self.conv1(x1_1)
        print("**feature Map1** shape:",featureMap1.shape) #[1,256,13,13] => 255인 이유

        '''feature Map 2 추출'''

        x1_2 = self.dbl2(x1)
        print(x1_2.shape) #[1, 512, 13, 13]
        x1_2 = self.upsample(x1_2)
        print(x1_2.shape) #[1, 512, 26, 26]
        x2 = torch.cat((x1_2, f2), dim=1)
        print(x2.shape) #[1, 1024, 26, 26]

        x2 = self.dbl1(self.dbl1(self.dbl1(self.dbl1(self.dbl1(x2)))))
        print("x2 shape:",  x2.shape) #[1,1024, 26,26]

        x2_1 = self.dbl2(x2) #[1,512, 26,26]
        featureMap2 = self.conv2(x2_1)
        print("**feature Map2** shape:",featureMap2.shape)  # [1,256,26,26] => 255인 이유

        '''feature Map 3 추출'''

        x2_2 = self.dbl5(x2)
        print("x2_2:", x2_2.shape)  # [1, 256, 26, 26]
        x2_2 = self.upsample(x2_2)
        print(x2_2.shape)  # [1, 256, 52, 52]
        x3 = torch.cat((x2_2, f3), dim=1)
        print(x3.shape)  # [1, 512, 52, 52]

        x3 = self.dbl3(self.dbl3(self.dbl3(self.dbl3(self.dbl3(x3)))))
        print(x3.shape) #[1, 512, 52, 52]

        x3 = self.dbl4(x3)
        print(x3.shape) #[1, 256, 52, 52]
        featureMap3 = self.conv3(x3)
        print(featureMap3.shape) #[1, 256, 52, 52]

        return featureMap1, featureMap2, featureMap3

if __name__ == '__main__':
    input_image = torch.randn(1, 3, 416, 416, dtype=torch.float)
    model = YOLOv3()
    f1,f2,f3 = model.forward(input_image)