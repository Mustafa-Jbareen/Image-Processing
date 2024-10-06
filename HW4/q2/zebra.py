# Mostufa Jbareen, 212955587
# Mohammed Egbaria, 318710761

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

image_path = 'zebra.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Section a
# Calculating fourier transform
fourier_spectrum = fftshift(fft2(image))
fourier_spectrum_disp = np.log(1 + np.absolute(fourier_spectrum))

# Section b
# Creating the image(2h,2w) by creating fourier(2h,2w) so that the fourier of original image * 4 is in the center
# and zero padding, so that we maintain the frequency and the colors of the original image in the new 2x image.
h, w = image.shape
fourier_spectrum_zero_padding = np.zeros((2 * h, 2 * w), dtype=complex)
fourier_spectrum_zero_padding[h // 2: 3 * h // 2, w // 2: 3 * w // 2] = fourier_spectrum * 4
fourier_spectrum_zero_padding_disp = np.log(1 + np.absolute(fourier_spectrum_zero_padding))
two_times_larger_grayscale_image = np.abs(ifft2(ifftshift(fourier_spectrum_zero_padding)))

# Section c
# Creating a*b copies of the image in one image(a*h,b*w), by modifying fourier transform in specific way,
# so that we have those copies.
a = b = 2
fourier_spectrum_four_copies = np.zeros((a * h, b * w), dtype=complex)
fourier_spectrum_four_copies[::a, ::b] = fourier_spectrum * abs(a * b)
fourier_spectrum_four_copies_disp = np.log(1 + np.absolute(fourier_spectrum_four_copies))
four_copies_grayscale_image = np.abs(ifft2(ifftshift(fourier_spectrum_four_copies)))

plt.figure(figsize=(10, 10))
plt.subplot(321)
plt.title('Original Grayscale Image')
plt.imshow(image, cmap='gray')

plt.subplot(322)
plt.title('Fourier Spectrum')
plt.imshow(fourier_spectrum_disp.astype("uint8"), cmap='gray')

plt.subplot(323)
plt.title('Fourier Spectrum Zero Padding')
plt.imshow(fourier_spectrum_zero_padding_disp.astype("uint8"), cmap='gray')

plt.subplot(324)
plt.title('Two Times Larger Grayscale Image')
plt.imshow(two_times_larger_grayscale_image.astype("uint8"), cmap='gray')

plt.subplot(325)
plt.title('Fourier Spectrum Four Copies')
plt.imshow(fourier_spectrum_four_copies_disp.astype("uint8"), cmap='gray')

plt.subplot(326)
plt.title('Four Copies Grayscale Image')
plt.imshow(four_copies_grayscale_image.astype("uint8"), cmap='gray')

plt.savefig('zebra_scaled.png')
plt.show()
