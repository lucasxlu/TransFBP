import os

import numpy as np
import pandas as pd


def mkdirs_if_not_exist(dir_name):
    """
    make directory if not exist
    :param dir_name:
    :return:
    """
    if not os.path.isdir(dir_name) or not os.path.exists(dir_name):
        os.makedirs(dir_name)


def out_result(test_set_filenames, predicted_list, gt_lst, attribute_list, path="./result/SCUT-FBP-TestSet.csv"):
    """
    output a Excel file containing testset filenames, predicted scores and groundtruth scores
    :param path:
    :param test_set_filenames:
    :param predicted_list:
    :param gt_lst:
    :return:
    """
    if attribute_list is not None:
        col = ['filename', 'predicted', 'groundtruth', 'attribute']
        arr = np.array([test_set_filenames, predicted_list, gt_lst, attribute_list])
    elif attribute_list is None:
        col = ['filename', 'predicted', 'groundtruth']
        arr = np.array([test_set_filenames, predicted_list, gt_lst])
    df = pd.DataFrame(arr.T, columns=col)
    mkdirs_if_not_exist('./result/')
    df.to_csv(path, index=False, encoding='UTF-8')
