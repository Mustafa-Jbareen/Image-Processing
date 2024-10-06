# Mostufa Jbareen, 212955587
# Mohammed Egbaria, 318710761

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore")


# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
    img_size = imgs_arr[0].shape
    res = []

    for img in imgs_arr:
        X = img.reshape(img_size[0] * img_size[1], 1)
        km = KMeans(n_clusters=n_colors)
        km.fit(X)

        img_compressed = km.cluster_centers_[km.labels_]
        img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

        res.append(img_compressed.reshape(img_size[0], img_size[1]))

    return np.array(res)


# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
    image_arrays = []
    lst = [file for file in os.listdir(folder) if file.endswith(formats)]
    for filename in lst:
        file_path = os.path.join(folder, filename)
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_arrays.append(gray_image)
    return np.array(image_arrays), lst


# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
    # Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values will work
    x_pos = 70 + 40 * idx
    y_pos = 274
    while image[y_pos, x_pos] == 1:
        y_pos -= 1
    return 274 - y_pos


# Sections c, d

def compare_hist(src_image: np.ndarray, target: np.ndarray) -> bool:
    """
    checking if specific digit (target) is in the image by pattern matching
    using EMD between the accumulated histograms of target and the image, and
    with threshold of 260
    * by finding that if the digit is in the image then it must be in window[x=113][y=30],
    we built the function to check only this window instead of sweeping through
    the whole image
    :param src_image: source image, the image we search in
    :param target: digit's image that we are searching for
    :return: TRUE if we find the digit in the picture as topmost number, otherwise FALSE
    """

    # constants
    x_pos = 113
    y_pos = 30
    emd_threshold = 260

    # getting the target accumulated histogram
    target_height, target_width = target.shape[:2]
    target_hist = cv2.calcHist([target], [0], None, [256], [0, 256]).flatten()
    target_acc_hist = np.cumsum(target_hist)

    # getting the specific pixel [113][30] accumulated histogram
    windows = np.lib.stride_tricks.sliding_window_view(src_image, (target_height, target_width))
    window_hist = cv2.calcHist([windows[x_pos][y_pos]], [0], None, [256], [0, 256]).flatten()
    window_acc_hist = np.cumsum(window_hist)

    # calculating the emd between target and the specific pixel accumulated histograms
    emd = 0
    for it in range(0, len(window_acc_hist)):
        emd += abs(window_acc_hist[it] - target_acc_hist[it])

    if emd < emd_threshold:
        return True
    return False


# Sections a, b
images, names = read_dir('data')
numbers, _ = read_dir('numbers')

cv2.imshow(names[0], images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

# section d
for i in range(len(numbers) - 1, -1, -1):
    if compare_hist(images[0], numbers[i]):
        break

# section e
# quantizing the images to 3 colors then thresholding the images,
# we found the threshold value after examining the quantized image,
# everything <= 213 -> 1, and <= 213 -> 255 in order to get a better image to do the search on.
im = quantization(images, 3)
black = 1
white = 255
for i in range(0, len(im)):
    threshold = np.max(im[i])
    im[i] = np.where(im[i] < threshold, black, white)

# doing section f, g for all images in one go.
# section f
for i in range(len(im)):
    # creating list of 10 items, in each slot the height of the bar in pixels.
    pixel_heights_list = [get_bar_height(im[i], num) for num in range(len(numbers))]
    max_pixel_height = max(pixel_heights_list)

    # finding the maximum student number in the image
    max_height = 0
    for j in range(0, len(numbers)):
        if compare_hist(images[i], numbers[j]):
            max_height = j

    # section g
    # implementing the formula to get how many students in each bin.
    if max_height != 0:
        heights = [round(pixel_heights_list[num] * max_height / max_pixel_height) for num in range(len(numbers))]
    else:
        heights = [0 for num in range(len(numbers))]

    # exit()

    id = i

    # The following print line is what you should use when printing out the final result - the text version of each histogram, basically.

    print(f'Histogram {names[id]} gave {heights}')
