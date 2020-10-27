import torch
from torch import nn
import torch.nn.functional as F

input_image = torch.randn(1,3,300,300,dtype=torch.float)

class SSD(nn.Module):
    def __init__(self):
      super(SSD, self).__init__()
      self.vgg = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64,128,3,padding=1),nn.ReLU(),
        nn.Conv2d(128,128,3,padding=1),nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(128,256,3,padding=1),nn.ReLU(),
        nn.Conv2d(256,256,3,padding=1),nn.ReLU(),
        nn.Conv2d(256,256,3,padding=1),nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

        nn.Conv2d(256,512,3,padding=1),nn.ReLU(),
        nn.Conv2d(512,512,3,padding=1),nn.ReLU(),
        nn.Conv2d(512,512,3,padding=1)
      )
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      self.conv5 = nn.Conv2d(512,512,3,padding=1)
      self.pool5 =  nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
      self.conv6 = nn.Conv2d(512,1024,3,padding=6,dilation=6)
      self.conv7 = nn.Conv2d(1024,1024,1)
      #Auxiliary Convolutional Layers
      self.conv8_1 = nn.Conv2d(1024,256,1,padding =0)
      self.conv8_2 = nn.Conv2d(256,512,3, padding=1, stride =2)
      self.conv9_1 = nn.Conv2d(512,128,1,padding=0)
      self.conv9_2 = nn.Conv2d(128,256,3,padding=1,stride =2)
      self.conv10_1 = nn.Conv2d(256,128,1,padding=0)
      self.conv10_2 = nn.Conv2d(128,256,3,padding=0)
      self.conv11_1 = nn.Conv2d(256,128,1,padding=0)
      self.conv11_2 = nn.Conv2d(128,256,3,padding=0)


    def forward(self,x):
      print("input shape:",x.shape)
      x=self.vgg(x)
      print("featureMap1:",x.shape)
      x = self.pool(x) #(1,512,19,19)
      x = F.relu(self.conv5(F.relu(self.conv5(F.relu(self.conv5(x))))))
      x = self.pool5(x)
      x = F.relu(self.conv6(x))
      x = F.relu(self.conv7(x))
      print("featureMap2:",x.shape)
      x=F.relu(self.conv8_2(self.conv8_1(x)))
      print("featureMap3:",x.shape)
      x=F.relu(self.conv9_2(self.conv9_1(x)))
      print("featureMap4:",x.shape)
      x=F.relu(self.conv10_2(self.conv10_1(x)))
      print("featureMap5:",x.shape)
      x=F.relu(self.conv11_2(self.conv11_1(x)))
      print("featureMap6:",x.shape)
      return x
model=SSD()
print((model.forward(input_image)).shape)