import math
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from skimage import io
from sklearn import decomposition
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

sys.path.append('../')
from config.cfg import config
from util.vgg_face_feature import extract_feature, extract_conv_feature


def split_train_and_test_data():
    """
    extract facial features and split it into train and test set
    :return:
    :version:1.0
    """
    df = pd.read_excel(config['label_excel_path'], 'Sheet1')
    filename_indexs = df['Image']
    attractiveness_scores = df['Attractiveness label']

    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * config['test_ratio'])
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    trainset_filenames = filename_indexs.iloc[train_indices]
    trainset_label = attractiveness_scores.iloc[train_indices]
    testset_filenames = filename_indexs.iloc[test_indices]
    testset_label = attractiveness_scores.iloc[test_indices]

    # extract Deep Features
    train_set_vector = [np.concatenate((extract_feature(config['face_image_filename'].format(_), layer_name='conv5_1'),
                                        extract_feature(config['face_image_filename'].format(_), layer_name='conv4_1')),
                                       axis=0) for _ in trainset_filenames]
    test_set_vector = [np.concatenate((extract_feature(config['face_image_filename'].format(_), layer_name='conv5_1'),
                                       extract_feature(config['face_image_filename'].format(_), layer_name='conv4_1')),
                                      axis=0) for _ in testset_filenames]

    # from facescore.features import RAW
    # train_set_vector = [RAW(config['face_image_filename'].format(_)) for _ in trainset_filenames]
    # test_set_vector = [RAW(config['face_image_filename'].format(_)) for _ in testset_filenames]

    # df = pd.DataFrame(np.array(train_set_vector + test_set_vector))
    # df.to_excel("./deep_feature.xlsx", sheet_name='features', index=False)
    #
    # df = pd.DataFrame(np.array(trainset_label + testset_label))
    # df.to_excel("./labels.xlsx", sheet_name='labels', index=False)

    return train_set_vector, test_set_vector, trainset_label, testset_label


def prepare_data():
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


def det_landmarks(image_path):
    """
    detect faces in one image, return face bbox and landmarks
    :param image_path:
    :return:
    """
    import dlib
    predictor = dlib.shape_predictor(config['predictor_path'])
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(image_path)
    faces = detector(img, 1)

    result = {}
    if len(faces) > 0:
        for k, d in enumerate(faces):
            shape = predictor(img, d)
            result[k] = {"bbox": [d.left(), d.top(), d.right(), d.bottom()],
                         "landmarks": [[shape.part(i).x, shape.part(i).y] for i in range(68)]}

    return result


def det_mat_landmarks(image):
    """
    detect faces image MAT, return face bbox and landmarks
    :param image_path:
    :return:
    """
    import dlib
    predictor = dlib.shape_predictor(config['predictor_path'])
    detector = dlib.get_frontal_face_detector()
    faces = detector(image, 1)

    result = {}
    if len(faces) > 0:
        for k, d in enumerate(faces):
            shape = predictor(image, d)
            result[k] = {"bbox": [d.left(), d.top(), d.right(), d.bottom()],
                         "landmarks": [[shape.part(i).x, shape.part(i).y] for i in range(68)]}

    return result


def PCA(feature_matrix, num_of_components=20):
    """
    PCA algorithm
    :param num_of_components:
    :param feature_matrix:
    :return:
    """
    pca = decomposition.PCA(n_components=num_of_components)
    pca.fit(feature_matrix)

    return pca.transform(feature_matrix)


def detect_face_and_cal_beauty(face_filepath='./talor.jpg'):
    """
    face detection with dlib
    :param face_filepath:
    :return:
    :version:1.0
    """
    import dlib

    print('start scoring your face...')
    # if the pre-trained model did not exist, then we train it from scratch
    if not os.path.exists(config['scut_fbp_reg_model']):
        print('No pre-trained model exists, start training now...')
        train_set_vector, test_set_vector, trainset_label, testset_label = split_train_and_test_data()
        train_model(train_set_vector, test_set_vector, trainset_label, testset_label)

    br = joblib.load(config['scut_fbp_reg_model'])
    print('Finishing training...')

    result = det_landmarks(face_filepath)

    image = cv2.imread(face_filepath)
    detector = dlib.get_frontal_face_detector()
    img = io.imread(face_filepath)

    dets, scores, idx = detector.run(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Left: {} Top: {} Right: {} Bottom: {} score: {} face_type:{}".format(d.left(), d.top(), d.right(),
                                                                                    d.bottom(), scores[i], idx[i]))
        roi = cv2.resize(image[d.top(): d.bottom(), d.left():d.right(), :],
                         (config['image_size'], config['image_size']),
                         interpolation=cv2.INTER_CUBIC)

        feature = np.concatenate(
            (extract_conv_feature(roi, layer_name='conv5_1'), extract_conv_feature(roi, layer_name='conv4_1')), axis=0)
        attractiveness = br.predict(feature.reshape(-1, feature.shape[0]))

        for index, face in result.items():
            # cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 225), 2)
            cv2.rectangle(image, (face['bbox'][0], face['bbox'][1]), (face['bbox'][2], face['bbox'][3]), (0, 255, 225),
                          2)
            cv2.putText(image, str(round(attractiveness[0] * 20, 2)), (d.left() + 5, d.top() - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (106, 106, 255), 0, cv2.LINE_AA)

            for ldmk in face['landmarks']:
                cv2.circle(image, (ldmk[0], ldmk[1]), 2, (255, 245, 0), -1)

        cv2.imwrite('tmp.jpg', image)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return round(attractiveness[0] * 20, 2), image


def train_model(train_set, test_set, train_label, test_label):
    """
    train ML model and serialize it into a binary pickle file
    :param train_set:
    :param test_set:
    :param train_label:
    :param test_label:
    :return:
    :Version:1.0
    """
    reg = linear_model.BayesianRidge()
    reg.fit(train_set, train_label)

    predicted_label = reg.predict(test_set)
    mae_lr = round(mean_absolute_error(test_label, predicted_label), 4)
    rmse_lr = round(math.sqrt(mean_squared_error(test_label, predicted_label)), 4)
    pc = round(np.corrcoef(test_label, predicted_label)[0, 1], 4)
    print('===============The Mean Absolute Error of Model is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Model is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Model is {0}===================='.format(pc))

    mkdirs_if_not_exist('./model')
    joblib.dump(reg, config['scut_fbp_reg_model'])
    print('The regression model has been persisted...')
    csv_tag = time.time()
    df = pd.DataFrame([mae_lr, rmse_lr, pc])
    out_result(test_set, predicted_label, test_label, None, path='./result/%f.csv' % csv_tag)
    df.to_csv('./result/performance_%s.csv' % csv_tag, index=False)
    print('The result csv file has been generated...')


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
    joblib.dump(reg, config['scut_fbp_reg_model'])
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


def mkdirs_if_not_exist(dir_name):
    """
    make directory if not exist
    :param dir_name:
    :return:
    """
    if not os.path.isdir(dir_name) or not os.path.exists(dir_name):
        os.makedirs(dir_name)


if __name__ == '__main__':
    # train_set, test_set = eccv_train_and_test_set(config['eccv_dataset_split_csv_file'])
    # train_and_eval_eccv(train_set, test_set)

    """
    split_csvs = [
        '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/eccv2010_split%d.csv' % _
        for _ in range(4, 6, 1)]
    for each_split in split_csvs:
        train_set, test_set = eccv_train_and_test_set(each_split)
        train_and_eval_eccv(train_set, test_set)
        sys.stdout.flush()
        time.sleep(3)
        print('*' * 100)
    """

    # cross validation
    # dataset, label = prepare_data()
    # cv_train(dataset, label)

    for idx in range(5):
        train_set_vector, test_set_vector, trainset_label, testset_label = split_train_and_test_data()
        train_model(train_set_vector, test_set_vector, trainset_label, testset_label)

    # detect_face_and_cal_beauty('./talor.jpg')

    # lbp = LBP('/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-48.jpg')
    # hog = HOG('/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-39.jpg')  # 512-d
    # harr = HARRIS('/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-39.jpg')
