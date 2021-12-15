import numpy as np
from PIL import Image

from SRCNN_classes import transform_class

image_path = './dataset/butterfly_GT.bmp'

kernel_size = [9,1,5]

image = Image.open(image_path)
image = np.array(image).astype(np.float32)

transform = transform_class(kernel_size=kernel_size)
crop_transform = transform.crop(image.shape[0],image.shape[0])

crop_image = crop_transform(image)
crop_image = np.array(crop_image).transpose([1,2,0]).astype(np.uint8)
crop_image = Image.fromarray(crop_image)
crop_image.save('./dataset/butterfly_GT_crop.bmp')