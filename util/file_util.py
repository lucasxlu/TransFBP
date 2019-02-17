import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.append('../')
from util.cfg import config
from util.vgg_face_feature import extract_feature


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


def prepare_scutfbp():
    """
    return the dataset and correspondent labels
    :return:
    """
    df = pd.read_excel(config['label_excel_path'], 'Sheet1')
    filename_indexs = df['Image']
    attractiveness_scores = df['Attractiveness label']

    dataset = [np.concatenate((extract_feature(config['face_image_filename'].format(_), layer_name='conv5_1'),
                               extract_feature(config['face_image_filename'].format(_), layer_name='conv4_1')),
                              axis=0) for _ in filename_indexs]

    return dataset, attractiveness_scores


def prepare_scutfbp5500(feat_layers):
    """
    prepare train/test set for SCUT-FBP5500 dataset
    :param feat_layers: features in these layer with concatenation fusion
    :return:
    """
    if not os.path.exists('../util/features') or len(os.listdir('../util/features')) == 0:
        print('No deep features...')
        sys.exit(0)

    train_face_img = pd.read_csv(os.path.abspath(os.path.join(config['scut_fbp5500_img_base_dir'], os.pardir,
                                                              'train_test_files/split_of_60%training and 40%testing/train.txt')),
                                 sep=' ', header=None).iloc[:, 0].tolist()
    train_score = pd.read_csv(os.path.abspath(os.path.join(config['scut_fbp5500_img_base_dir'], os.pardir,
                                                           'train_test_files/split_of_60%training and 40%testing/train.txt')),
                              sep=' ', header=None).iloc[:, 1].astype(np.float).tolist()

    test_face_img = pd.read_csv(os.path.abspath(os.path.join(config['scut_fbp5500_img_base_dir'], os.pardir,
                                                             'train_test_files/split_of_60%training and 40%testing/test.txt')),
                                sep=' ', header=None).iloc[:, 0].tolist()
    test_score = pd.read_csv(os.path.abspath(os.path.join(config['scut_fbp5500_img_base_dir'], os.pardir,
                                                          'train_test_files/split_of_60%training and 40%testing/test.txt')),
                             sep=' ', header=None).iloc[:, 1].astype(np.float).tolist()

    train_feats = []
    test_feats = []

    print('loading serialized deep features...')
    for _ in train_face_img:
        with open('../util/features/{0}.pkl'.format(_.split('.')[0]), mode='rb') as f:
            feat = []
            layer_feat_dict = pickle.load(f)
            for k, v in layer_feat_dict.items():
                if k in feat_layers:
                    feat += np.array(layer_feat_dict[k]).ravel().tolist()
        train_feats.append(feat)

    for _ in test_face_img:
        with open('../util/features/{0}.pkl'.format(_.split('.')[0]), mode='rb') as f:
            feat = []
            layer_feat_dict = pickle.load(f)
            for k, v in layer_feat_dict.items():
                if k in feat_layers:
                    feat += np.array(layer_feat_dict[k]).ravel().tolist()
        test_feats.append(feat)

    return np.array(train_feats), np.array(train_score), np.array(test_feats), np.array(test_score)
