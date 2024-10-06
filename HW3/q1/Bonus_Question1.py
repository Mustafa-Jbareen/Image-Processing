# Mostufa Jbareen, 212955587
# Mohammed Egbaria, 318710761

import cv2
import numpy as np


# Answers:
# 1 -> average filter, param = (10000,1), MSE = 0.4
# 2 -> average filter, param = (11, 11), MSE = 2.4
# 3 -> median filter, param = 11, MSE = 0.4
# 4 -> gaussian filter, param = (1, 15), 20, MSE = 2.1
# 5 -> High pass filter (sharpen filter + 128), MSE = 3.8
# 6 -> laplacian filter, MSE = 6.3
# 7 -> split and swap, MSE = 1.2
# 8 -> the same image, MSE = 0.3
# 9 -> sharpened image, MSE = 24.1

def mse_error(first_image: np.array, second_image: np.array):
    print(f"MSE: {np.sum((first_image - second_image) ** 2) / (first_image.shape[0] * first_image.shape[1]):.1f}")
    return


def average_filter(img: np.array, kernal):
    return cv2.blur(img, kernal)


def median_filter(img: np.array):
    return cv2.medianBlur(img, 11)


def laplacian_filter(img: np.array):
    laplacian_kernel = np.array([[-0.3, -0.6, -0.3],
                                 [0, 0, 0],
                                 [0.3, 0.6, 0.3]], dtype=np.float64)
    return cv2.convertScaleAbs(cv2.filter2D(img, -1, laplacian_kernel))


def high_pass_photo(img: np.array):
    return img - cv2.bilateralFilter(img, 13, 255, 255) + 128


def display_and_save(img: np.array, _id):
    cv2.imshow(f"image {_id} Recreation", img)
    cv2.waitKey(0)
    cv2.imwrite(f"image_{_id}_Recreation.jpg", img)


def gaussian_filter(img: np.array):
    return cv2.GaussianBlur(img, (1, 15), 20)


def bilateral_filter(img: np.array):
    return cv2.bilateralFilter(img, 11, 50, 50)


def sharpen_image(img: np.array):
    b = img.astype(np.float64)

    bb = cv2.bilateralFilter(img, 5, 100, 16).astype(np.float64)
    s = b - bb
    s = np.where(s >= 0, s, 0)

    bg = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float64)
    k = b - bg
    k = np.where(k >= 0, k, 0)

    return cv2.addWeighted(b.astype("uint8"), 1, s.astype("uint8"), 1.9, 0)


def split_and_swap(img: np.array):
    """
    split the image in half and swap two halves
    :param img:
    :return: image split in half and swapped
    """
    n, m = img.shape[:2]
    kernal = np.zeros((n, m), dtype=np.float64)
    kernal[n // 2, m // 2] = 1
    padded_img = cv2.copyMakeBorder(img, n // 2, n // 2, 0, 0, cv2.BORDER_WRAP)
    return cv2.filter2D(padded_img, 0, kernal)[n::]


if __name__ == "__main__":
    # reading data
    image = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
    image_1 = cv2.imread("image_1.jpg", cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread("image_2.jpg", cv2.IMREAD_GRAYSCALE)
    image_3 = cv2.imread("image_3.jpg", cv2.IMREAD_GRAYSCALE)
    image_4 = cv2.imread("image_4.jpg", cv2.IMREAD_GRAYSCALE)
    image_5 = cv2.imread("image_5.jpg", cv2.IMREAD_GRAYSCALE)
    image_6 = cv2.imread("image_6.jpg", cv2.IMREAD_GRAYSCALE)
    image_7 = cv2.imread("image_7.jpg", cv2.IMREAD_GRAYSCALE)
    image_8 = cv2.imread("image_8.jpg", cv2.IMREAD_GRAYSCALE)
    image_9 = cv2.imread("image_9.jpg", cv2.IMREAD_GRAYSCALE)

    # image 1
    display_and_save(average_filter(image, (10000, 1)), 1)
    mse_error(average_filter(image, (10000, 1)), image_1)

    # image 2
    display_and_save(average_filter(image, (11, 11)), 2)
    mse_error(image_2, average_filter(image, (11, 11)))

    # image 3
    display_and_save(median_filter(image), 3)
    mse_error(image_3, median_filter(image))

    # image 4
    display_and_save(gaussian_filter(image), 4)
    mse_error(image_4, gaussian_filter(image))

    # image 5
    display_and_save(high_pass_photo(image), 5)
    mse_error(image_5, high_pass_photo(image))

    # image 6
    display_and_save(laplacian_filter(image), 6)
    mse_error(image_6, laplacian_filter(image))

    # image 7
    display_and_save(split_and_swap(image), 7)
    mse_error(split_and_swap(image), image_7)

    # image 8
    display_and_save(image, 8)
    mse_error(image_8, image)

    # image 9
    display_and_save(sharpen_image(image), 9)
    mse_error(image_9, sharpen_image(image))
