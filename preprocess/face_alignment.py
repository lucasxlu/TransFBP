import os
import sys

import cv2
import numpy as np
import tensorflow as tf

sys.path.append('../')
from main.face_beauty_regressor import det_landmarks, det_mat_landmarks


def face_align(face_image_path):
    img = cv2.imread(face_image_path)
    result = det_landmarks(face_image_path)

    face = result.get(0)

    left_eye_left_ldmk = face['landmarks'][36]
    left_eye_right_ldmk = face['landmarks'][39]

    right_eye_left_ldmk = face['landmarks'][42]
    right_eye_right_ldmk = face['landmarks'][45]

    left_center = [int((left_eye_left_ldmk[0] + left_eye_right_ldmk[0]) / 2),
                   int((left_eye_left_ldmk[1] + left_eye_right_ldmk[1]) / 2)]

    right_center = [int((right_eye_left_ldmk[0] + right_eye_right_ldmk[0]) / 2),
                    int((right_eye_left_ldmk[1] + right_eye_right_ldmk[1]) / 2)]

    theta = np.degrees(np.arctan((right_center[1] - left_center[1]) / (right_center[0] - left_center[0])))
    print('theta = %f ...' % theta)

    face_bbox = face['bbox']

    rows, cols, chs = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    roi = dst[face_bbox[0]: face_bbox[2], face_bbox[1]: face_bbox[3], :]

    roi = cv2.resize(roi, (128, 128))

    rst = det_mat_landmarks(roi)
    f = rst.get(0)

    l_eye_l_ldmk = f['landmarks'][36]
    l_eye_r_ldmk = f['landmarks'][39]

    r_eye_l_ldmk = f['landmarks'][42]
    r_eye_r_ldmk = f['landmarks'][45]

    l_eye_center = [(l_eye_l_ldmk[0] + l_eye_r_ldmk[0]) / 2,
                    (l_eye_l_ldmk[1] + l_eye_r_ldmk[1]) / 2]

    r_eye_center = [(r_eye_l_ldmk[0] + r_eye_r_ldmk[0]) / 2,
                    (r_eye_l_ldmk[1] + r_eye_r_ldmk[1]) / 2]

    center_x = (l_eye_center[0] + r_eye_center[0]) / 2
    center_y = (l_eye_center[1] + r_eye_center[1]) / 2

    k = 40.0 / l_eye_center[0]
    print('The scale ratio k = %d ...' % k)
    face_region = cv2.resize(roi, (int(k * 128), int(k * 128)))

    # cv2.imwrite('/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/face/%s' %
    #             face_image_path.split('/')[-1], face_region)

    cv2.imshow('roi', roi)
    cv2.imshow('face_region', face_region)
    cv2.waitKey()
    cv2.destroyAllWindows()


def det_lean_degree(face_image_path):
    result = det_landmarks(face_image_path)

    face = result.get(0)

    left_eye_left_ldmk = face['landmarks'][36]
    left_eye_right_ldmk = face['landmarks'][39]

    right_eye_left_ldmk = face['landmarks'][42]
    right_eye_right_ldmk = face['landmarks'][45]

    left_center = [int((left_eye_left_ldmk[0] + left_eye_right_ldmk[0]) / 2),
                   int((left_eye_left_ldmk[1] + left_eye_right_ldmk[1]) / 2)]

    right_center = [int((right_eye_left_ldmk[0] + right_eye_right_ldmk[0]) / 2),
                    int((right_eye_left_ldmk[1] + right_eye_right_ldmk[1]) / 2)]

    theta = np.degrees(np.arctan((right_center[1] - left_center[1]) / (right_center[0] - left_center[0])))

    return theta


def rotate_point_by_theta(point, center=(0, 0), theta=0):
    """
    rotate a point around center by theta degree
    :param point:
    :param center:
    :param theta:
    :return:
    """
    x_new = (point[0] - center[0]) * np.cos(theta) - (point[1] - center[1]) * np.sin(theta) + center[0]
    y_new = (point[0] - center[0]) * np.sin(theta) + (point[1] - center[1]) * np.cos(theta) + center[1]

    return tuple(x_new, y_new)


def split_lean_or_not_data(image_dir):
    fail_list = []
    for imagefile in os.listdir(image_dir):
        abs_file = os.path.join(image_dir, imagefile)
        try:
            theta = det_lean_degree(abs_file)
            if np.abs(theta) < 5:
                print('{0} is normal, degree = {1}'.format(imagefile, theta))
            else:
                print('{0} is leaned, degree = {1}'.format(imagefile, theta))
        except:
            fail_list.append(imagefile)

    print(len(fail_list))


if __name__ == '__main__':
    # face_align(
    #     "/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/hotornot_face/female_18_A8HMBSR_face_1.jpg")

    hotornot_face_dir = '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/hotornot_face/'
    aligned_hotornot_face_dir = '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/aligned_hotornot_face/'
    lean_hotornot_face_dir = '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/lean_hotornot_face/'

    all_face_list = os.listdir(hotornot_face_dir)
    aligned_face_list = os.listdir(aligned_hotornot_face_dir)

    print(aligned_face_list)
    print(aligned_hotornot_face_dir)

    for _ in all_face_list:
        if _ not in aligned_face_list:
            tf.gfile.Copy(os.path.join(hotornot_face_dir, _), os.path.join(lean_hotornot_face_dir, _), overwrite=True)
