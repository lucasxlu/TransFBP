import os

import numpy as np
import cv2

SCUT_FBP = '/media/lucasx/Document/DataSet/Face/SCUT-FBP/Data_Collection/Data_Collection'
IMAGE_SIZE = 224


def padding_images(dst_dir='/media/lucasx/Document/DataSet/Face/SCUT-FBP/Padding'):
    if not os.path.exists(dst_dir) or not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    for _ in os.listdir(SCUT_FBP):
        image = cv2.imread(os.path.join(SCUT_FBP, _))
        h, w, c = image.shape
        dst = np.ones([IMAGE_SIZE, IMAGE_SIZE, c], dtype=np.uint8) * 255
        if h >= w:
            ratio = h / IMAGE_SIZE
            roi = cv2.resize(image, (int(w / ratio), IMAGE_SIZE))
            padding_width = int((IMAGE_SIZE - int(w / ratio)) / 2)
            dst[:, padding_width:padding_width + int(w / ratio), :] = roi
        else:
            ratio = h / IMAGE_SIZE
            roi = cv2.resize(image, (IMAGE_SIZE, int(h / ratio)))
            padding_height = int((IMAGE_SIZE - int(h / ratio)) / 2)
            dst[padding_height:padding_height + int(h / ratio), :, :] = roi

        cv2.imwrite(os.path.join(dst_dir, _), dst)
        print('write image %s' % os.path.join(dst_dir, _))


def warp_images(dst_dir='/media/lucasx/Document/DataSet/Face/SCUT-FBP/Warp'):
    if not os.path.exists(dst_dir) or not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    for _ in os.listdir(SCUT_FBP):
        image = cv2.imread(os.path.join(SCUT_FBP, _))
        re_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        cv2.imwrite(os.path.join(dst_dir, _), re_image)
        print('write image %s' % os.path.join(dst_dir, _))


if __name__ == '__main__':
    padding_images()
