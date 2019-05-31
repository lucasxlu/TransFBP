import numpy as np
import cv2
import sys
import time

from sklearn import linear_model

sys.path.append('../')
from util.vgg_face_feature import extract_conv_feature


def infer_from_img(img_file):
    img = cv2.imread(img_file)
    img = cv2.resize(img, (224, 224))
    # tik = time.time()
    feat = extract_conv_feature(img, layer_name='conv5_1').tolist()
    reg = linear_model.BayesianRidge(np.array(feat))
    reg.fit(np.random.rand(1, len(feat)), np.array(1))
    tik = time.time()
    score = reg.predict(feat)
    tok = time.time()

    print('Beauty score is {0}, it takes {1} seconds!'.format(score, (tok - tik) * 1000))


if __name__ == '__main__':
    infer_from_img('../testimg.jpg')
