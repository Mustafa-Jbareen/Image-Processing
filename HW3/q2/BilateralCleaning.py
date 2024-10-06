# Mostufa Jbareen, 212955587
# Mohammed Egbaria, 318710761

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    """
    This function cleans the Gaussian noise using Bilateral Cleaning technique
    :param im: the noised image
    :param radius: the radius of the window
    :param stdSpatial: the std of the Gaussian window used for the spatial weight
    :param stdIntensity:the std of the Gaussian window used for the intensity weight
    :return: Cleared version of the image im
    """
    # Preparing the filtered image
    cleaned = np.zeros((len(im), len(im[0])))

    # Creating the ranges of x and y for calculating gs
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, y)

    # since gs is same for all pixels we calculate it outside the loop
    gs = np.exp(-((xx ** 2) + (yy ** 2)) / (2 * (stdSpatial ** 2)))

    # Running on all pixels and behave accordingly
    for i in range(len(cleaned)):
        for j in range(len(cleaned[0])):
            # Getting y and x ranges.
            x_indices = np.arange(i - radius, i + radius + 1)
            y_indices = np.arange(j - radius, j + radius + 1)

            # Ensuring indices are within image bounds
            x_indices = np.clip(x_indices, 0, im.shape[0] - 1)
            y_indices = np.clip(y_indices, 0, im.shape[1] - 1)

            # Creating meshgrid of indices
            xx, yy = np.meshgrid(x_indices, y_indices)

            # Extracting the window from the image
            window = im[xx, yy].astype(np.float64)

            # Calculating gi
            gi = np.exp(-(((window - im[i, j]) ** 2) / (2 * (stdIntensity ** 2))))

            # Updating the value of the cleaned image
            cleaned[i, j] = np.sum(gs * gi * window) / np.sum(gs * gi)

    # Returning the filtered image
    return cleaned.astype("uint8")


# This function reads the image
def read(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# This function displays the image and the cleaned image using pyplot
def display(img, cleaned_img):
    plt.subplot(121)
    plt.imshow(img, cmap='gray')

    plt.subplot(122)
    plt.imshow(cleaned_img, cmap='gray')

    plt.show()
    return None


# Clearing taj image
image = read('taj.jpg')
std_spatial = 2 / 100 * np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
std_intensity = 30
clear_image_b = clean_Gaussian_noise_bilateral(image, 7, std_spatial, std_intensity)
display(image, clear_image_b)
cv2.imwrite("cleaned_taj.jpg", clear_image_b)

# Clearing NoisyGrayImage image
image = read('NoisyGrayImage.png')
std_spatial = 2 / 100 * np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
std_intensity = 90
clear_image_b = clean_Gaussian_noise_bilateral(image, 7, std_spatial, std_intensity)
display(image, clear_image_b)
cv2.imwrite("cleaned_NoisyGrayImage.jpg", clear_image_b)

# Clearing balls image
image = read('balls.jpg')
std_spatial = 2 / 100 * np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
std_intensity = 7
clear_image_b = clean_Gaussian_noise_bilateral(image, 7, std_spatial, std_intensity)
display(image, clear_image_b)
cv2.imwrite("cleaned_balls.jpg", clear_image_b)
