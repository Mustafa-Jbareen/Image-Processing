# Mostufa Jbareen, 212955587
# Mohammed Egbaria, 318710761

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import convolve

matplotlib.use('TkAgg')


def get_gaussian_pyramid(image, levels):
    """
    this function takes an image and returns the gaussain pyramid of the image
    :param image: grey-scale image
    :param levels: the levels of the pyramid
    :return: list contains the layers of the gaussian pyramid of the image
    """
    pyramid = [image]
    current_layer = image
    for _ in range(levels - 1):
        current_layer = cv2.GaussianBlur(current_layer, (7, 7), 0)
        current_layer = current_layer[::2, ::2]
        pyramid.append(current_layer)

    return pyramid


# The same function from Q1
def scale_down(image, resize_ratio):
    fourier_spectrum = fftshift(fft2(image))
    h, w = image.shape
    new_h, new_w = int(h / resize_ratio), int(w / resize_ratio)

    start_row = (h - new_h) // 2
    start_col = (w - new_w) // 2
    end_row = start_row + new_h
    end_col = start_col + new_w

    cropped_spectrum = fourier_spectrum[start_row:end_row, start_col:end_col]

    return np.abs(ifft2(ifftshift(cropped_spectrum)))


# The same function from Q2
def scale_up(image, resize_ratio):
    fourier_spectrum = fftshift(fft2(image))
    h, w = image.shape
    new_h, new_w = int(h * resize_ratio), int(w * resize_ratio)
    fourier_spectrum_zero_padding = np.zeros((new_h, new_w), dtype=complex)

    start_row = (new_h - h) // 2
    start_col = (new_w - w) // 2
    end_row = start_row + h
    end_col = start_col + w

    fourier_spectrum_zero_padding[start_row:end_row, start_col:end_col] = fourier_spectrum

    return np.abs(ifft2(ifftshift(fourier_spectrum_zero_padding))) * (resize_ratio ** 2)


def get_laplacian_pyramid(image, levels):
    """
    this function takes an image and levels, and it returns the laplacian pyramid of the image as a list,
    it does that by applying what we learned in the class, and by using the gaussian pyramid, and a simple formula
    we learned in class
    :param image: the image we want to create the laplacian pyramid for
    :param levels: levels of the pyramid
    :return: the laplacian pyramid of the given image
    """
    gaussian_pyramid = get_gaussian_pyramid(image, levels)
    laplacian_pyramid = []
    # Running on the levels and doing the needed
    for i in range(levels - 1):
        expanded = scale_up(gaussian_pyramid[i + 1], 2)
        laplacian_pyramid.append(gaussian_pyramid[i] - expanded)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # Append the smallest level

    return laplacian_pyramid


def restore_from_pyramid(pyramidList, resize_ratio=2):
    """
    this function gets a Laplacian pyramid and returns the image by "collapsing" the pyramid as we saw in the class
    :param pyramidList: laplacian pyramid of certain image
    :param resize_ratio: default resize ratio = 2, because each layer is 2x larger than the layer above it
    :return: the image (restored from the laplacian pyramid)
    """
    curr = pyramidList[len(pyramidList) - 1]
    # Running on the layers and "collapsing" the pyramid
    for i in range(len(pyramidList) - 2, -1, -1):
        temp = scale_up(curr, resize_ratio)
        temp += pyramidList[i]
        curr = temp

    return curr


def validate_operation(img):
    pyr = get_laplacian_pyramid(img, levels)
    img_restored = restore_from_pyramid(pyr)

    plt.title(f"MSE is {np.mean((img_restored - img) ** 2)}")
    plt.imshow(img_restored, cmap='gray')

    plt.show()


def blend_pyramids(levels):
    """
    this function uses two images laplacian pyramids in order to blends two images into one image but in smart way,
    it blends each layer in the laplacian pyramid in specific cross dissolve in the middle of the image,
    which give us a smooth transition between the two images
    :param levels: the levels of the pyramids
    :return: laplacian pyramid of one blended image that in left side image1(orange) is dominant ,
    and in left side image2(apple) is dominant with a smooth transition between the two sides in the middle
    """
    blend_pyr = []
    # Running on the levels
    for curr_level in range(levels):
        # creating the mask of the cross-dissolve
        mask = np.zeros(pyr_apple[curr_level].shape)
        width = mask.shape[1]

        # Initialize mask's columns
        mask[:, :int((0.5 * width) - (curr_level + 1))] = 1.0

        # Applying the given cross-dissolve formula
        for i in range(2 * (curr_level + 1)):
            mask[:, (width // 2) - (curr_level + 1) + i] = 0.9 - 0.9 * i / (2 * (curr_level + 1))

        # Adding the layer to the blended image laplacian pyramid
        blend_pyr.append((pyr_orange[curr_level] * mask) + (pyr_apple[curr_level] * (1 - mask)))

    return blend_pyr


apple = cv2.imread('apple.jpg')
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

orange = cv2.imread('orange.jpg')
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

levels = 3

validate_operation(orange)
validate_operation(apple)

# Creating laplacian pyramids for both images Orange and Apple
pyr_apple = get_laplacian_pyramid(apple, levels)
pyr_orange = get_laplacian_pyramid(orange, levels)

# Blending Pyramids
pyr_result = blend_pyramids(levels)

# Getting and plotting the blended image
final = restore_from_pyramid(pyr_result)
plt.imshow(final, cmap='gray')
plt.show()

cv2.imwrite(f"result.jpg", final)
