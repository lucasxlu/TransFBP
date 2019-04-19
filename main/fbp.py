import math
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import dask.dataframe as dd
import dask.array as da

sys.path.append('../')
from util.calc_util import split_train_and_test_data
from util.cfg import config
from util.file_util import mkdirs_if_not_exist, out_result, prepare_scutfbp5500
from util.vgg_face_feature import extract_feature


def train(train_set, test_set, train_label, test_label, data_name, test_filenames, distribute_training=False):
    """
    train ML model and serialize it into a binary pickle file
    :param train_set:
    :param test_set:
    :param train_label:
    :param test_label:
    :param data_name
    :param test_filenames
    :param distribute_training
    :return:
    :Version:1.0
    """
    print("The shape of training set is {0}".format(np.array(train_set).shape))
    print("The shape of test set is {0}".format(np.array(test_set).shape))
    print('Use distribute training ? >> {0}'.format(distribute_training))
    reg = linear_model.BayesianRidge()
    if not distribute_training:
        reg.fit(train_set, train_label)
    else:
        train_set, test_set, train_label, test_label = da.array(train_set), da.array(test_set), da.array(
            train_label), da.array(test_label)
        reg.fit(train_set, train_label)

    predicted_label = reg.predict(test_set)
    mae_lr = round(mean_absolute_error(test_label, predicted_label), 4)
    rmse_lr = round(math.sqrt(mean_squared_error(test_label, predicted_label)), 4)
    pc = round(np.corrcoef(test_label, predicted_label)[0, 1], 4)
    print('===============The Mean Absolute Error of Model is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Model is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Model is {0}===================='.format(pc))

    mkdirs_if_not_exist('./model')
    joblib.dump(reg, './model/BayesRidge_%s.pkl' % data_name)
    print('The regression model has been persisted...')

    mkdirs_if_not_exist('./result')

    out_result(test_filenames, predicted_label, test_label, None, path='./result/Pred_GT_{0}.csv'.format(data_name))

    df = pd.DataFrame([mae_lr, rmse_lr, pc])
    df.to_csv('./result/%s.csv' % data_name, index=False)
    print('The result csv file has been generated...')


def cv_train(dataset, labels, cv=10):
    """
    train model with cross validation
    :param model:
    :param dataset:
    :param labels:
    :param cv:
    :return:
    """
    reg = linear_model.BayesianRidge()
    mae_list = -cross_val_score(reg, dataset, labels, cv=cv, n_jobs=-1, scoring='neg_mean_absolute_error')
    rmse_list = np.sqrt(-cross_val_score(reg, dataset, labels, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error'))
    pc_list = cross_val_score(reg, dataset, labels, cv=cv, n_jobs=-1, scoring='r2')

    print(mae_list)
    print(rmse_list)
    print(pc_list)

    print('=========The Mean Absolute Error of Model is {0}========='.format(np.mean(mae_list)))
    print('=========The Root Mean Square Error of Model is {0}========='.format(np.mean(rmse_list)))
    print('=========The Pearson Correlation of Model is {0}========='.format(np.mean(pc_list)))

    mkdirs_if_not_exist('./model')
    joblib.dump(reg, "./model/BayesRidge_SCUT-FBP.pkl")
    print('The regression model has been persisted...')


def eccv_train_and_test_set(split_csv_filepath):
    """
    split train and test eccv dataset
    :param split_csv_filepath:
    :return:
    :Version:1.0
    """
    df = pd.read_csv(split_csv_filepath, header=None)
    filenames = [os.path.join(os.path.dirname(split_csv_filepath), 'hotornot_face', _.replace('.bmp', '.jpg')) for _ in
                 df.iloc[:, 0].tolist()]
    scores = df.iloc[:, 1].tolist()
    flags = df.iloc[:, 2].tolist()

    train_set = dict()
    test_set = dict()

    for i in range(len(flags)):
        if flags[i] == 'train':
            train_set[filenames[i]] = scores[i]
        elif flags[i] == 'test':
            test_set[filenames[i]] = scores[i]

    return train_set, test_set


def eccv_train_and_test_set_with_align_or_lean(split_csv_filepath):
    """
    split train and test eccv dataset with judging alignment and lean
    :param split_csv_filepath:
    :return:
    :Version:1.0
    """
    df = pd.read_csv(split_csv_filepath, header=None)
    filenames = [os.path.join(os.path.dirname(split_csv_filepath), 'hotornot_face', _.replace('.bmp', '.jpg')) for _ in
                 df.iloc[:, 0].tolist()]
    scores = df.iloc[:, 1].tolist()
    flags = df.iloc[:, 2].tolist()

    from preprocess.eccv_face_attribute_preprocess import load_eccv_attribute
    eccv_attribute = load_eccv_attribute('./eccv_face_attribute.csv')

    aligned_train_set = dict()
    aligned_test_set = dict()
    lean_train_set = dict()
    lean_test_set = dict()

    for i in range(len(flags)):
        if flags[i] == 'train':
            if eccv_attribute[filenames[i].split('/')[-1]] == 'aligned':
                aligned_train_set[filenames[i]] = scores[i]
            elif eccv_attribute[filenames[i].split('/')[-1]] == 'lean':
                lean_train_set[filenames[i]] = scores[i]
        elif flags[i] == 'test':
            if eccv_attribute[filenames[i].split('/')[-1]] == 'aligned':
                aligned_test_set[filenames[i]] = scores[i]
            elif eccv_attribute[filenames[i].split('/')[-1]] == 'lean':
                lean_test_set[filenames[i]] = scores[i]

    return aligned_train_set, aligned_test_set, lean_train_set, lean_test_set


def train_and_eval_eccv(train, test):
    """
    train and test eccv dataset
    :param train:
    :param test:
    :return:
    """
    train_vec = list()
    train_label = list()
    test_vec = list()
    test_label = list()

    for k, v in train.items():
        feature = np.concatenate((extract_feature(k, layer_name="conv5_2"), extract_feature(k, layer_name="conv5_3")),
                                 axis=0)
        train_vec.append(feature)
        train_label.append(v)

    for k, v in test.items():
        feature = np.concatenate((extract_feature(k, layer_name="conv5_2"), extract_feature(k, layer_name="conv5_3")),
                                 axis=0)
        test_vec.append(feature)
        test_label.append(v)

    reg = linear_model.BayesianRidge()
    reg.fit(np.array(train_vec), np.array(train_label))
    mkdirs_if_not_exist('./model')
    joblib.dump(reg, config['eccv_fbp_reg_model'])

    predicted_label = reg.predict(np.array(test_vec))
    mae_lr = round(mean_absolute_error(np.array(test_label), predicted_label), 4)
    rmse_lr = round(math.sqrt(mean_squared_error(np.array(test_label), predicted_label)), 4)
    pc = round(np.corrcoef(test_label, predicted_label)[0, 1], 4)

    print('===============The Mean Absolute Error of Model is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Model is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Model is {0}===================='.format(pc))

    csv_tag = time.time()

    mkdirs_if_not_exist('./result')
    df = pd.DataFrame([mae_lr, rmse_lr, pc])
    df.to_csv('./result/performance_%s.csv' % csv_tag, index=False)

    out_result(list(test.keys()), predicted_label.flatten().tolist(), test_label, None,
               path='./result/detail_%s.csv' % csv_tag)


def train_and_eval_eccv_with_align_or_lean(aligned_train, aligned_test, lean_train, lean_test):
    """
    train and eval model with frontal faces and side faces
    :param aligned_train:
    :param aligned_test:
    :param lean_train:
    :param lean_test:
    :return:
    """
    aligned_train_vec = list()
    aligned_train_label = list()
    aligned_test_vec = list()
    aligned_test_label = list()

    lean_train_vec = list()
    lean_train_label = list()
    lean_test_vec = list()
    lean_test_label = list()

    test_filenames = list()
    attribute_list = list()

    for k, v in aligned_train.items():
        feature = np.concatenate((extract_feature(k, layer_name="conv5_2"), extract_feature(k, layer_name="conv5_3")),
                                 axis=0)
        aligned_train_vec.append(feature)
        aligned_train_label.append(v)

    for k, v in aligned_test.items():
        test_filenames.append(k)
        attribute_list.append('aligned')

        feature = np.concatenate((extract_feature(k, layer_name="conv5_2"), extract_feature(k, layer_name="conv5_3")),
                                 axis=0)
        aligned_test_vec.append(feature)
        aligned_test_label.append(v)

    for k, v in lean_train.items():
        feature = np.concatenate((extract_feature(k, layer_name="conv5_2"), extract_feature(k, layer_name="conv5_3")),
                                 axis=0)
        lean_train_vec.append(feature)
        lean_train_label.append(v)

    for k, v in lean_test.items():
        test_filenames.append(k)
        attribute_list.append('lean')

        feature = np.concatenate((extract_feature(k, layer_name="conv5_2"), extract_feature(k, layer_name="conv5_3")),
                                 axis=0)
        lean_test_vec.append(feature)
        lean_test_label.append(v)

    aligned_reg = linear_model.BayesianRidge()
    lean_reg = linear_model.BayesianRidge()

    aligned_reg.fit(np.array(aligned_train_vec), np.array(aligned_train_label))
    lean_reg.fit(np.array(lean_train_vec), np.array(lean_train_label))
    mkdirs_if_not_exist('./model')
    joblib.dump(aligned_reg, './model/eccv_fbp_dcnn_bayes_reg_aligned.pkl')
    joblib.dump(lean_reg, './model/eccv_fbp_dcnn_bayes_reg_lean.pkl')

    aligned_predicted_label = aligned_reg.predict(np.array(aligned_test_vec))
    lean_predicted_label = lean_reg.predict(np.array(lean_test_vec))

    predicted_label = aligned_predicted_label.tolist() + lean_predicted_label.tolist()
    test_label = aligned_test_label + lean_test_label

    mae_lr = round(mean_absolute_error(np.array(test_label), np.array(predicted_label)), 4)
    rmse_lr = round(math.sqrt(mean_squared_error(np.array(test_label), np.array(predicted_label))), 4)
    pc = round(np.corrcoef(test_label, predicted_label)[0, 1], 4)

    aligned_mae_lr = round(
        mean_absolute_error(np.array(aligned_test_label), np.array(aligned_predicted_label.tolist())), 4)
    aligned_rmse_lr = round(
        math.sqrt(mean_squared_error(np.array(aligned_test_label), np.array(aligned_predicted_label.tolist()))), 4)
    aligned_pc = round(np.corrcoef(aligned_test_label, aligned_predicted_label.tolist())[0, 1], 4)

    lean_mae_lr = round(mean_absolute_error(np.array(lean_test_label), np.array(lean_predicted_label)), 4)
    lean_rmse_lr = round(math.sqrt(mean_squared_error(np.array(lean_test_label), np.array(lean_predicted_label))), 4)
    lean_pc = round(np.corrcoef(lean_test_label, lean_predicted_label)[0, 1], 4)

    print('===============The Mean Absolute Error of Model is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Model is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Model is {0}===================='.format(pc))

    mkdirs_if_not_exist('./result')

    csv_file_tag = time.time()

    df = pd.DataFrame([mae_lr, rmse_lr, pc])
    df.to_csv('./result/%f_all.csv' % csv_file_tag, index=False)

    aligned_df = pd.DataFrame([aligned_mae_lr, aligned_rmse_lr, aligned_pc])
    aligned_df.to_csv('./result/%f_aligned.csv' % csv_file_tag, index=False)

    lean_df = pd.DataFrame([lean_mae_lr, lean_rmse_lr, lean_pc])
    lean_df.to_csv('./result/%f_lean.csv' % csv_file_tag, index=False)

    out_result(test_filenames, predicted_label, test_label, attribute_list, './result/%f_detail.csv' % csv_file_tag)


def train_and_eval_scutfbp(train_set_vector, test_set_vector, trainset_label, testset_label, testset_filenames):
    """
    train and eval on SCUT-FBP dataset
    :param train_set_vector:
    :param test_set_vector:
    :param trainset_label:
    :param testset_label:
    :param testset_filenames
    :return:
    """
    print("The shape of training set is {0}".format(np.array(train_set_vector).shape))
    print("The shape of test set is {0}".format(np.array(test_set_vector).shape))
    reg = linear_model.BayesianRidge()
    reg.fit(train_set_vector, trainset_label)

    predicted_label = reg.predict(test_set_vector)
    mae_lr = round(mean_absolute_error(testset_label, predicted_label), 4)
    rmse_lr = round(math.sqrt(mean_squared_error(testset_label, predicted_label)), 4)
    pc = round(np.corrcoef(testset_label, predicted_label)[0, 1], 4)
    print('===============The Mean Absolute Error of Model is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Model is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Model is {0}===================='.format(pc))

    mkdirs_if_not_exist('./model')
    joblib.dump(reg, './model/BayesRidge_SCUTFBP.pkl')
    print('The regression model has been persisted...')

    mkdirs_if_not_exist('./result')

    out_result(testset_filenames, predicted_label, testset_label, None, path='./result/Pred_GT_SCUTFBP.csv')

    df = pd.DataFrame([mae_lr, rmse_lr, pc])
    df.to_csv('./result/BayesRidge_SCUTFBP.csv', index=False)
    print('The result csv file has been generated...')


if __name__ == '__main__':
    # train and test on HotOrNot
    # train_set, test_set = eccv_train_and_test_set(config['eccv_dataset_split_csv_file'])
    # train_and_eval_eccv(train_set, test_set)

    # train and test on SCUT-FBP
    train_set_vector, test_set_vector, trainset_label, testset_label, trainset_filenames, testset_filenames = split_train_and_test_data()
    train_and_eval_scutfbp(train_set_vector, test_set_vector, trainset_label, testset_label, testset_filenames)

    # train and test on SCUT-FBP5500
    # train_feats, train_score, test_feats, test_score, train_filenames, test_filenames = prepare_scutfbp5500(feat_layers=["conv4_1", "conv5_1"])
    # train(train_feats, test_feats, train_score, test_score, "SCUT-FBP5500", distribute_training=True)
