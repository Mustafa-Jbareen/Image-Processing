# Mostufa Jbareen, 212955587
# Mohammed Egbaria, 318710761

import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import warnings

matplotlib.use('TkAgg')
warnings.filterwarnings("ignore")


# Section a
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


# Section b
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


# Section c
def ncc_2d(im, patt):
    windows = np.lib.stride_tricks.sliding_window_view(im, patt.shape)
    ncc_im = np.zeros(windows.shape[:2])
    patt_mean = np.mean(patt)
    patt_var = np.sum((pattern - patt_mean) ** 2)

    # implementing the formula for calculating ncc
    for row in range(len(windows)):
        for col in range(len(windows[0])):
            window_mean = np.mean(windows[row, col])
            window_var = np.sum((windows[row, col] - window_mean) ** 2)
            means_sum = np.sum((windows[row, col] - window_mean) * (patt - patt_mean))
            denominator = np.sqrt(window_var * patt_var)
            if denominator == 0:
                continue
            ncc_im[row, col] = means_sum / denominator

    return ncc_im


def display(image, pattern):
    plt.subplot(2, 3, 1)
    plt.title('Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Pattern')
    plt.imshow(pattern, cmap='gray', aspect='equal')

    ncc = ncc_2d(image, pattern)

    # max_index = np.argmax(ncc)
    # max_coordinates = np.unravel_index(max_index, ncc.shape)
    # print(max_coordinates)

    plt.subplot(2, 3, 5)
    plt.title('Normalized Cross-Correlation Heatmap')
    plt.imshow(ncc ** 2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')

    cbar = plt.colorbar()
    cbar.set_label('NCC Values')

    plt.show()


def draw_matches(image, matches, pattern_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for point in matches:
        y, x = point
        top_left = (int(x - pattern_size[1] / 2), int(y - pattern_size[0] / 2))
        bottom_right = (int(x + pattern_size[1] / 2), int(y + pattern_size[0] / 2))
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)

    plt.imshow(image, cmap='gray')
    plt.show()

    cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image)


def find_matches(im, threshold):
    matches = []
    for i in range(len(im)):
        for j in range(len(im[0])):
            if im[i, j] >= threshold:
                matches.append([i, j])
    return np.array(matches)


CURR_IMAGE = "students"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

############# Students #############
image_scale_ratio = 2
pattern_scale_ratio = 1
threshold = 0.55

image_scaled = scale_up(image, image_scale_ratio)  # Your code goes here. If you choose not to scale the image, just remove it.
patten_scaled = scale_down(pattern, pattern_scale_ratio)  # Your code goes here. If you choose not to scale the pattern, just remove it.

# section d
display(image_scaled, patten_scaled)

ncc = ncc_2d(image_scaled, patten_scaled)  # Your code goes here
real_matches = find_matches(ncc, threshold)  # Your code goes here

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:, 0] += patten_scaled.shape[0] // 2			# if pattern was not scaled, replace this with "pattern"
real_matches[:, 1] += patten_scaled.shape[1] // 2			# if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.
real_matches[:, 0] = (real_matches[:, 0] // image_scale_ratio).astype(np.int32)
real_matches[:, 1] = (real_matches[:, 1] // image_scale_ratio).astype(np.int32)

draw_matches(image, real_matches, pattern.shape)  # if pattern was not scaled, replace this with "pattern"

# ############# Crew #############
CURR_IMAGE = "thecrew"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
image_scale_ratio = 4
pattern_scale_ratio = 1
threshold = 0.4

image_scaled = scale_up(image, image_scale_ratio)  # Your code goes here. If you choose not to scale the image, just remove it.
patten_scaled = scale_up(pattern, pattern_scale_ratio)  # Your code goes here. If you choose not to scale the pattern, just remove it.

# section d
display(image_scaled, patten_scaled)

ncc = ncc_2d(image_scaled, patten_scaled)  # Your code goes here
real_matches = find_matches(ncc, threshold)  # Your code goes here

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:, 0] += patten_scaled.shape[0] // 2  # if pattern was not scaled, replace this with "pattern"
real_matches[:, 1] += patten_scaled.shape[1] // 2  # if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.
real_matches[:, 0] //= image_scale_ratio
real_matches[:, 1] //= image_scale_ratio

draw_matches(image, real_matches, (pattern.shape[0]//2, pattern.shape[1]//2))  # if pattern was not scaled, replace this with "pattern"