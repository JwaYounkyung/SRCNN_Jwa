import argparse

import torch
import torch.backends.cudnn as cudnn
from torch import nn

import torchvision

import numpy as np
import PIL.Image as pil_image

from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

kernel_size = [9,1,5]
### SRCNN model
class SRCNN(nn.Module):
    def __init__(self, num_channels=1): # in_channels : RGB or YCbCr
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default="results/model_weight.pth")
    parser.add_argument('--image-file', type=str, default="dataset/butterfly_GT.bmp")
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
    image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    crop_transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(243)
    ])

    y = ycbcr[..., 0]
    y /= 255.
    # y = crop_transform(y).to(device)
    # y = y.unsqueeze(0)
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    crop_y = torch.squeeze(y,0)
    crop_y = crop_transform(crop_y)
    crop_y = crop_y.unsqueeze(0)

    psnr = calc_psnr(crop_y, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))
