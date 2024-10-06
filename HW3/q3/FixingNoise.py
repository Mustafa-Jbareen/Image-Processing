# Mostufa Jbareen, 212955587
# Mohammed Egbaria, 318710761

import cv2
import numpy as np


def first_section():
    # Reading the data
    image = cv2.imread("broken.jpg", cv2.IMREAD_GRAYSCALE)
    image.astype(np.float64)

    # Median filtering
    median_filtered_image = cv2.medianBlur(image, 5)

    # Bilateral filtering on the median filtered image
    bi_med = cv2.bilateralFilter(median_filtered_image, 21, 10, 20)

    # Applying high pass filter
    bilateral_filtered_image = cv2.bilateralFilter(bi_med, 7, 20, 19)  # Bilateral filtering
    mask = bi_med.astype(np.float64) - bilateral_filtered_image.astype(np.float64)
    mask = np.where(mask >= 0, mask, 0).astype("uint8")
    result = cv2.addWeighted(bi_med, 1, mask, 1.5, 0).astype("uint8")

    # Displaying and saving the result
    cv2.imshow("sol", result)
    cv2.waitKey(0)

    cv2.imwrite("section_a_sol.jpg", result)


def second_section():
    # Reading the data.
    images = np.load("noised_images.npy", allow_pickle=True)
    res = np.zeros((images[0].shape[0], images[0].shape[1])).astype(np.float32)

    # Running on all images, and we do median filter on every image, and then we sharpen the filtered image,
    # and finally adding it to the result with the right ratio 1/200.
    for image in images:
        cleaned = cv2.medianBlur(image, 3).astype(np.float32)
        bilateral = cv2.bilateralFilter(cleaned, 7, 10, 10).astype(np.float32)
        diff = cleaned - bilateral
        cleaned = cv2.addWeighted(cleaned, 1, diff, 2, 0)
        res = cv2.addWeighted(cleaned, 1 / 200, res, 1, 0)
    cv2.imshow("Solution", res.astype("uint8"))
    cv2.waitKey(0)

    # Saving the solution
    cv2.imwrite("section_b_sol.jpg", res)


if __name__ == "__main__":
    first_section()
    second_section()
