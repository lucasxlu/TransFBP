# inference demo
import sys
import time
import os

import cv2
from sklearn.externals import joblib

sys.path.append('../')
from util.vgg_face_feature import extract_conv_feature

BAYES_RIDGE_REG_WEIGHTS = './model/BayesRidge_SCUT-FBP5500.pkl'
assert os.path.exists(BAYES_RIDGE_REG_WEIGHTS)


def infer_from_img(model, img_file):
    """
    infer from image with a given model
    """
    img = cv2.imread(img_file)
    img = cv2.resize(img, (224, 224))
    feat = extract_conv_feature(img, layer_name='fc6').tolist() + extract_conv_feature(img, layer_name='fc7').tolist()
    tik = time.time()
    score = model.predict(feat)
    tok = time.time()

    print('Beauty score is {0}, it takes {1} seconds!'.format(score, tok - tik))


if __name__ == '__main__':
    bayes_ridge_reg = joblib.load(BAYES_RIDGE_REG_WEIGHTS)
    infer_from_img(bayes_ridge_reg, '../testimg.jpg')
