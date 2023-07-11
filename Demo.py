import numpy as np
import cv2

def denoise_image(image, sigma):
  """Denoises an image using a Gaussian filter.

  Args:
    image: The image to denoise.
    sigma: The standard deviation of the Gaussian filter.

  Returns:
    The denoised image.
  """

  filtered_image = cv2.GaussianBlur(image, (3, 3), sigmaX=sigma, sigmaY=sigma)
  return filtered_image

if __name__ == "__main__":
  # Load the image.
  image = cv2.imread("MRI_noisy.tif")

  # Add noise to the image.
  noisy_image = image + np.random.normal(0, 0.1, image.shape)

  # Denoise the image.
  denoised_image = denoise_image(noisy_image, sigma=0.5)

  # Save the denoised image.
  cv2.imwrite("denoised_image.jpg", denoised_image)