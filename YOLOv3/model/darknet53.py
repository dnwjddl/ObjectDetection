'''
Darknet-53 for yolo v3
'''
import torch
from torch import nn

input_image = torch.randn(1,3,416,416,dtype=torch.float)

def dbl(in_num, out_num, kernel_size = 3, padding = 1, stride= 1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size= kernel_size, stride= stride, padding = padding, bias = False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU()
    )

class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()
        reduced_channels = int(in_channels/2)
        self.layer1 = dbl(in_channels,reduced_channels, kernel_size=1,padding=0)
        self.layer2 = dbl(reduced_channels, in_channels)

    def forward(self,x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

class Darknet53(nn.Module):
    def __init__(self, block):
        super(Darknet53, self).__init__()
        self.conv1 = dbl(3, 32)
        self.conv2 = dbl(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)

        self.conv3 = dbl(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)

        self.conv4 = dbl(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)

        self.conv5 = dbl(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)

        self.conv6 = dbl(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)

        #self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(1024, self.num_classes)

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        print("Conv1:",out.shape)
        out = self.conv2(out)
        print("Conv2:",out.shape)
        out = self.residual_block1(out)
        print("Res1:",out.shape)

        out = self.conv3(out)
        print("Conv3:",out.shape)
        out = self.residual_block2(out)
        print("Res2:",out.shape)

        out = self.conv4(out)
        print("Conv4:",out.shape)
        out = self.residual_block3(out)

        feature_map3 = out
        print("**feature_map3**:",feature_map3.shape)

        out = self.conv5(out)
        print("Conv5:",out.shape)
        out = self.residual_block4(out)
        feature_map2 = out
        print("**feature_map2**:",feature_map2.shape)

        out = self.conv6(out)
        print("Conv6:",out.shape)
        out = self.residual_block5(out)
        feature_map1 = out
        print("**feature_map1**:",feature_map1.shape)
        #out = self.global_avg_pool(out)
        #out = out.view(-1, 1024)
        #out = self.fc(out)
        print("==========DarkNet END===========")

        return feature_map1, feature_map2, feature_map3

if __name__ == "__main__":
    model = Darknet53(DarkResidualBlock)
    f1, f2, f3 = model.forward(input_image)
