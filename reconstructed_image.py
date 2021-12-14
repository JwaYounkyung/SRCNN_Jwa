### import
import torch

import numpy as np
from PIL import Image

from SRCNN_classes import SRCNN, transform_class, PSNR
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb

### device setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### parameter setting
model_path = './results/model_weight_915.pth'
image_path = './dataset/butterfly_GT.bmp'

kernel_size = [9,1,5]

### model load
model = SRCNN(kernel_size=kernel_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

### image load and convert to ycbcr
image = Image.open(image_path)
image = np.array(image).astype(np.float32)
ycbcr = convert_rgb_to_ycbcr(image)

### crop setting
transform = transform_class(kernel_size=kernel_size)
crop_only_transform = transform.crop_only(image.shape[0],image.shape[0])
crop_transform = transform.crop(image.shape[0],image.shape[0])

### PSNR
''' 코드 참조 부분'''
y = ycbcr[..., 0] 
y /= 255.
y = torch.from_numpy(y).to(device)
y = y.unsqueeze(0).unsqueeze(0)
''''''

crop_y = torch.squeeze(y,0)
crop_y = crop_only_transform(crop_y)
crop_y = crop_y.unsqueeze(0)

with torch.no_grad():
    preds = model(y)

psnr = PSNR(crop_y, preds)
print('PSNR of SRCNN model', psnr)

### crop cb and cr
cb = ycbcr[..., 1]
cb = crop_transform(cb)
cb = torch.squeeze(cb).cpu().numpy()

cr = ycbcr[..., 2]
cr = crop_transform(cr)
cr = torch.squeeze(cr).cpu().numpy()

### merge YCbCr
''' 코드 참조 부분 '''
preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

reconstructed_ycbcr = np.array([preds, cb, cr]).transpose([1,2,0]) # CxHxW to HxWxC
reconstructed_rgb = np.clip(convert_ycbcr_to_rgb(reconstructed_ycbcr), 0.0, 255.0).astype(np.uint8)
reconstructed_image = Image.fromarray(reconstructed_rgb)
reconstructed_image.save('./results/butterfly_GT_srcnn_x3.bmp')
''''''



