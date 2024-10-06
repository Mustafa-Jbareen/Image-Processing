# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import matplotlib.pyplot as plt
import numpy as np


def apply_fix(image: np.array, id: int) -> np.array:
    """
    applying some correction on image
    :param image: the image we need to activate correction on
    :param id: id of the image
    :return: image after some correction
    """

    if id == 1:
        return histogram_equalization(image)
    elif id == 2:
        return gamma_correction(image, 1 / 2.2)
    elif id == 3:
        return brightness_contrast_max_stretch(image)


def brightness_contrast_max_stretch(im: np.array) -> np.array:
    """
    max brightness contrast stretch as we saw in tutorial
    :param im: image
    :return: returning im after doing max brightness contrast stretch
    """
    minimum = np.min(im)
    maximum = np.max(im)
    old_contrast = maximum - minimum
    a = 255 / old_contrast
    b = - (255 / old_contrast) * minimum
    return cv2.convertScaleAbs(im, alpha=a, beta=b)


def gamma_correction(im: np.array, gamma: float) -> np.array:
    """
    gamma correction on specific image
    :param im: image
    :param gamma: hyperparameter
    :return: im after activating gamma correction on it
    """
    return np.uint8(np.power(image / 255.0, gamma) * 255.0)


def histogram_equalization(im: np.array) -> np.array:
    """
    returning im after equalizeHist.
    :param im: image
    :return: result of equalizeHist on im
    """
    return cv2.equalizeHist(im)


for i in range(1, 4):
    if i != 1:
        path = f'{i}.jpg'
    else:
        path = f'{i}.png'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    fixed_image = apply_fix(image, i)
    if i != 1:
        plt.imsave(f'(OPTIONAL) {i} - fixed.jpg', fixed_image, cmap='gray')
    else:
        plt.imsave(f'(OPTIONAL) {i} - fixed.png', fixed_image, cmap='gray')
