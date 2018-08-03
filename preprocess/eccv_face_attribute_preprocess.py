import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.append('../')
from preprocess.face_alignment import det_lean_degree

aligned_eccv_face_dir = '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/aligned_hotornot_face'
lean_eccv_face_dir = '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/lean_hotornot_face'


def remove_lean_face():
    for _ in os.listdir(aligned_eccv_face_dir):
        try:
            theta = det_lean_degree(os.path.join(aligned_eccv_face_dir, _))
            print('theta = %f' % theta)
            if 0 <= math.fabs(theta) <= 5:
                pass
            else:
                os.remove(os.path.join(aligned_eccv_face_dir, _))
        except:
            print('detect facial landmarks error!')


def lean_faces():
    aligned_list = [_ for _ in os.listdir(aligned_eccv_face_dir)]

    for _ in os.listdir(lean_eccv_face_dir):
        if _ in aligned_list:
            os.remove(os.path.join(lean_eccv_face_dir, _))
            print('remove %s' % _)

    print('processing done!')


def load_eccv_attribute(eccv_attribute_csv):
    """
    load eccv face attribute file
    :param eccv_attribute_csv:
    :return:
    """
    filenames = pd.read_csv(eccv_attribute_csv)['filename']
    attributes = pd.read_csv(eccv_attribute_csv)['attribute']

    return dict(zip(filenames, attributes))


if __name__ == '__main__':
    # load_eccv_attribute('./eccv_face_attribute.csv')

    lean_list = [_ for _ in os.listdir(lean_eccv_face_dir)]
    aligned_list = [_ for _ in os.listdir(aligned_eccv_face_dir)]

    attr_list = ['aligned' for i in range(len(aligned_list))] + ['lean' for i in range(len(lean_list))]

    col = ['filename', 'attribute']
    df = pd.DataFrame(np.array([aligned_list + lean_list, attr_list]).T, columns=col)
    df.to_csv('./eccv_face_attribute.csv', index=False)
