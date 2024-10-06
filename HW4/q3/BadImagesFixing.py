# Mostufa Jbareen, 212955587
# Mohammed Egbaria, 318710761

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage

matplotlib.use('TkAgg')


def clean_baby(im):
    # Applying Geometric Operation on the three images and after that cleaning the salt and pepper noise using
    # median filter, and enhancing the final result using gamma correction

    # Data preprocessing
    res = np.zeros((256, 256), dtype="uint8")
    corners = np.array([[0, 0], [255, 255], [255, 0], [0, 255]], dtype=np.float32)
    up_left = np.array([[6, 20], [111, 130], [111, 20], [6, 130]], dtype=np.float32)
    up_right = np.array([[181, 5], [176, 120], [249, 70], [121, 50]], dtype=np.float32)
    down_middle = np.array([[78, 162], [245, 160], [146, 117], [133, 244]], dtype=np.float32)
    matches = (up_left, up_right, down_middle)
    images = []

    # Applying Geometric Transformations
    for dst_points in matches:
        t = cv2.getPerspectiveTransform(corners, dst_points)
        p = cv2.warpPerspective(im, np.linalg.inv(t), (256, 256), flags=cv2.INTER_CUBIC)
        images.append(p)

    # Median Filtering Process
    images = np.array(images)
    res = np.median(images, axis=0).astype("uint8")
    res = cv2.medianBlur(res, 7)

    # Sharpen the result
    temp = cv2.GaussianBlur(res, (5, 5), 0)
    temp = res - temp
    temp = np.where(temp < 0, 0, temp).astype(np.float64)
    res = cv2.addWeighted(res.astype(np.float64), 1, temp.astype(np.float64), 1, 0).astype(np.uint8)

    # Enhancing the image using Gamma Correction
    res = np.power(res / 255.0, 0.8) * 255.0
    return res.astype(np.uint8)


def clean_windmill(im):
    # Clearing the windmill image by finding the peak manually and assigning it to 0
    fourier_transform = fft2(im)
    fourier_transform[4, 28] = 0
    fourier_transform[-4 % 256, -28 % 256] = 0
    return np.abs(ifft2(fourier_transform))


def clean_watermelon(im):
    # B-Sharp filter
    temp = cv2.GaussianBlur(im, (5, 5), 0)
    temp = im - temp
    return cv2.addWeighted(im.astype(np.float64), 1, temp.astype(np.float64), 6, 0).astype("uint8")


def clean_umbrella(im):
    # Cleaning the umbrella by finding the shifted convolution manually and thin dividing the
    # fourier of the noisy image by the fourier of the mask.
    im_fourier = fft2(im)
    mask = np.zeros(im.shape)
    mask[0, 0] = 0.5
    mask[4, 79] = 0.5
    mask_fourier = fft2(mask)

    # we take this step in order not to divide by 0
    mask_fourier = np.where(np.abs(mask_fourier) < 0.01, 1, mask_fourier)

    # Getting the deionised image fourier  by applying the formula we saw in Tutorial 7
    clear_fourier = im_fourier / mask_fourier
    return np.abs(ifft2(clear_fourier))


def clean_USAflag(im):
    # Applying median and horizontal average filter only on the stripes area of the flag
    res = im.copy()
    res[90:, 0:] = ndimage.median_filter(im[90:, 0:], [1, 15])
    res[90:, 0:] = cv2.blur(res[90:, 0:], (100, 1))
    res[:90, 143:] = ndimage.median_filter(im[:90, 143:], [1, 9])
    res[:90, 143:] = cv2.blur(res[:90, 143:], (100, 1))
    return res


def clean_house(im):
    # Cleaning the house by finding the shifted convolution manually and thin dividing the
    # fourier of the noisy image by the fourier of the mask.
    im_fourier = fft2(im)
    mask = np.zeros(im.shape)
    mask[0, 0:10] = 0.1
    mask_fourier = fft2(mask)

    # we take this step in order not to divide by 0
    mask_fourier = np.where(np.abs(mask_fourier) < 0.01, 1, mask_fourier)

    # Getting the deionised image fourier  by applying the formula we saw in Tutorial 7
    clear_fourier = im_fourier / mask_fourier
    cleared_image = np.abs(ifft2(clear_fourier))

    # Removing any irrelevant data
    cleared_image = np.where(cleared_image < 0, 0, cleared_image)
    cleared_image = np.where(cleared_image > 255, 255, cleared_image)
    return cleared_image


def clean_bears(im):
    # Applying max brightness and contrast on the image
    minimum = np.min(im)
    maximum = np.max(im)
    old_contrast = maximum - minimum
    a = 255 / old_contrast
    b = - (255 / old_contrast) * minimum
    return cv2.convertScaleAbs(im, alpha=a, beta=b)
