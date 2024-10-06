# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import os
import shutil
import sys

import numpy.linalg


# section c
def get_transform(matches, is_affine):
    """
    calculating the transform from img-1 to img-k
    :param matches: matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1,kp2) where kpi = (x,y)
    :param is_affine: the type of the transformation
    :return: t - the transformation from img-1 to img-k
    """
    # separating matches data to dst_point = image-1 points, and src_points = image-k points
    src_points, dst_points = matches[:, 0], matches[:, 1]
    src_points = src_points.astype(np.float32)
    dst_points = dst_points.astype(np.float32)

    # checking the type of the transformation, and we react according to it and returning the transformation
    if is_affine:
        t = cv2.getAffineTransform(src_points, dst_points)
    else:
        t = cv2.getPerspectiveTransform(src_points, dst_points)
    return t


# section e
def stitch(img1, img2):
    """
    averaging img1, and img2 using cv2.addWeighted(img1, 0.5, img2, 0.5, 0), then adding
    0.5 * (where img1 and img2 do not overlap) by
    cv2.addWeighted(cv2.addWeighted(img1, 0.5, img2, 0.5, 0), 1, cv2.absdiff(img1, img2), 0.5, 0)
    and that is because when we averaged on img1 and img2, we also averaged on the places they do not overlap,
    which mean  img1 = 0.5 * img1 and img2 = 0.5 * img2 in those places, so we add them back
    :param img1: current puzzle progress
    :param img2: new image we add stitch to puzzle
    :return: puzzle after stitching imag2
    """
    return cv2.addWeighted(cv2.addWeighted(img1, 0.5, img2, 0.5, 0), 1, cv2.absdiff(img1, img2), 0.5, 0)


# section d
def inverse_transform_target_image(target_img, original_transform, output_size):
    """
    calculating and returning the transformed target_image (in it's right place in the puzzle) (target_image_absolute),
    using cv2.warpPerspective() if the transform is perspective or if cv2.warpAffine() the transform is affine
    :param target_img: image-k
    :param original_transform: the transform from image-1 to image-k
    :param output_size: (image 1 width image 1h height)
    :return: image-k_absolute (in it's right place in the puzzle)
    """

    # checking if the transform is affine or perspective by checking original transform shape (3,3) | (2,3)
    if original_transform.shape == (3, 3):
        return cv2.warpPerspective(target_img, numpy.linalg.inv(transform), (output_size[1], output_size[0]),
                                   flags=cv2.INTER_LINEAR)
    else:
        return cv2.warpAffine(target_img, cv2.invertAffineTransform(transform), (output_size[1], output_size[0]),
                              flags=cv2.INTER_LINEAR)


# returns list of pieces file names
def prepare_puzzle(puzzle_dir):
    edited = os.path.join(puzzle_dir, 'abs_pieces')
    if os.path.exists(edited):
        shutil.rmtree(edited)
    os.mkdir(edited)

    affine = 4 - int("affine" in puzzle_dir)

    matches_data = os.path.join(puzzle_dir, 'matches.txt')
    n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

    matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images - 1, affine, 2, 2)

    return matches, affine == 3, n_images


if __name__ == '__main__':
    # puzzle names list
    lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']

    # running on and solving all puzzles
    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')

        # preparing paths
        puzzle = os.path.join('puzzles', puzzle_dir)
        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')

        # section b
        matches, is_affine, n_images = prepare_puzzle(puzzle)

        # reading 1'st piece, which is placed correctly
        piece_1_path = os.path.join(pieces_pth, "piece_1.jpg")
        piece_1 = cv2.imread(piece_1_path)

        # saving piece_1_absolute, which is given to us.
        cv2.imwrite(os.path.join(edited, f"piece_1_absolute.jpg"), piece_1)

        # final_puzzle = the solved puzzle
        # we initialize the final_puzzle to be the piece_1, and we stitch the other pieces to it.
        final_puzzle = piece_1

        # running on all puzzle pieces except the first one because it is already placed right.
        for k in range(2, n_images + 1):
            # calculating the transformation from piece_1 to piece_k
            transform = get_transform(matches[k - 2, :, :], is_affine)

            # reading the piece_k
            piece_k_path = os.path.join(pieces_pth, f"piece_{k}.jpg")
            piece_k = cv2.imread(piece_k_path)

            # calculating piece_k_absolute using inverse_transform_target_image function
            im_k_absolute = inverse_transform_target_image(piece_k, transform, piece_1.shape)

            # saving piece_k_absolute in the abs_pieces folder
            cv2.imwrite(os.path.join(edited, f"piece_{k}_absolute.jpg"), im_k_absolute)

            # stitching piece_k_absolute to our current puzzle progress using stitch function
            final_puzzle = stitch(final_puzzle, im_k_absolute)

        # section f
        # outputting the solution
        cv2.imshow("Solution of" + puzzle, final_puzzle)
        cv2.waitKey(0)

        # saving puzzle solution in the right folder
        sol_file = f'solution.jpg'
        cv2.imwrite(os.path.join(puzzle, sol_file), final_puzzle)
