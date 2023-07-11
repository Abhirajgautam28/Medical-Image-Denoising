import numpy as np
import cv2

# load the medical image
img = cv2.imread('MRI_noisy.tif')

# apply Non-Local Means denoising
denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# save the denoised image
cv2.imwrite('denoised_image.png', denoised)
