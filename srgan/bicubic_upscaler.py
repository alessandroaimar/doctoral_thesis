# Compute bicubic upscaled picture

from PIL import Image
import numpy as np

test_image_hr_path = './srgan/images/test_images/original_highres.png'
test_image_lr_path = './srgan/images/test_images/original_lowres.png'
output_image_path  = './srgan/images/test_images/bicubic_highres.png'

test_image_lr = Image.open(test_image_lr_path).convert("RGB")
test_image_hr = Image.open(test_image_hr_path).convert("RGB")

img_upscaled =  test_image_lr.resize((test_image_lr.size[0]*4, test_image_lr.size[1]*4),Image.BICUBIC)
img_upscaled.save(output_image_path)

print

# Compute mean squared error
mse = np.mean((np.array(test_image_hr, dtype=np.float32) - np.array(img_upscaled, dtype=np.float32)) ** 2) 
max_pixel = 255
psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 


print ("PSNR Bicubic: {}".format(psnr))
