import os
import sys

from skimage import io
from sklearn import decomposition
from sklearn.externals import joblib

import cv2
import numpy as np
import pandas as pd

sys.path.append('../')
from util.vgg_face_feature import extract_feature, extract_conv_feature
from util.cfg import config


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
    if not os.path.exists("./model/BayesRidge_SCUT-FBP.pkl"):
        print('No pre-trained model exists, start training now...')
        sys.exit(0)

    br = joblib.load("./model/BayesRidge_SCUT-FBP.pkl")
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

    return train_set_vector, test_set_vector, trainset_label, testset_label
