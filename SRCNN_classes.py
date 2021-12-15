import torch
from torch import nn
import torchvision

### SRCNN model
class SRCNN(nn.Module):
    def __init__(self, num_channels=1, kernel_size=[9,1,5]): # num_channels : RGB or YCbCr
        super(SRCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=kernel_size[0], stride=1, padding=0),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=kernel_size[1], stride=1, padding=0),
            nn.ReLU())

        self.layer3 = nn.Conv2d(in_channels=32, out_channels=num_channels, kernel_size=kernel_size[2], stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class transform_class:
    def __init__(self, kernel_size=[9,1,5]):
        self.transform_crop = None
        self.kernel_size = kernel_size
    
    def crop(self, fsub1, fsub2):
        self.crop_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), # HxWxC to CxHxW
            torchvision.transforms.CenterCrop((fsub1-sum(self.kernel_size)+3,fsub2-sum(self.kernel_size)+3)) # fsub-f1-f2-f3+3 | 33-9-1-5+3
        ])
        return self.crop_transform
    
    def crop_only(self, fsub1, fsub2):
        self.crop_transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((fsub1-sum(self.kernel_size)+3,fsub2-sum(self.kernel_size)+3))
        ])
        return self.crop_transform

def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10.0 * torch.log10(1.0 / mse)
