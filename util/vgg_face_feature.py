import sys
import os
import pickle

import tensorflow as tf
from scipy.misc import imread, imresize
import json

sys.path.append('../')
from model.vgg_face import vgg_face
from util.cfg import config


def extract_feature(image_filepath, layer_name='conv5_1'):
    graph = tf.Graph()
    with graph.as_default():
        input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
        output, average_image, class_names = vgg_face(config['vgg_face_model_mat_file'], input_maps)

    # read sample image
    img = imread(image_filepath, mode='RGB')
    img = (img - img.mean()) / img.std()
    img = imresize(img, [224, 224])

    # run the graph
    with tf.Session(graph=graph) as sess:
        [out] = sess.run([output], feed_dict={input_maps: [img]})
        feature = out[layer_name]
        sess.close()

    return feature.ravel()


def get_fm(image_filepath, layer_name='conv5_1'):
    """
    get feature map
    :param image_filepath:
    :param layer_name:
    :return:
    """
    graph = tf.Graph()
    with graph.as_default():
        input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
        output, average_image, class_names = vgg_face(config['vgg_face_model_mat_file'], input_maps)

    # read sample image
    img = imread(image_filepath, mode='RGB')
    img = (img - img.mean()) / img.std()
    img = imresize(img, [224, 224])

    # run the graph
    with tf.Session(graph=graph) as sess:
        [out] = sess.run([output], feed_dict={input_maps: [img]})
        feature = out[layer_name]
        sess.close()

    return feature


def extract_conv_feature(img, layer_name='conv5_1'):
    graph = tf.Graph()
    with graph.as_default():
        input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
        output, average_image, class_names = vgg_face(config['vgg_face_model_mat_file'], input_maps)

    img = (img - img.mean()) / img.std()
    img = imresize(img, [224, 224])

    # run the graph
    with tf.Session(graph=graph) as sess:
        [out] = sess.run([output], feed_dict={input_maps: [img]})
        feature = out[layer_name]
        sess.close()

    return feature.ravel()


def extract_deep_feats(img_fname):
    """
    extract deep features and save its features to a pickle file
    :param img_fname:
    :return:
    """
    if not os.path.exists('./features'):
        os.makedirs('./features')

    img = imread(img_fname, mode='RGB')
    img = (img - img.mean()) / img.std()
    img = imresize(img, [224, 224])

    graph = tf.Graph()
    with graph.as_default():
        input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
        output, average_image, class_names = vgg_face(config['vgg_face_model_mat_file'], input_maps)

    img = (img - img.mean()) / img.std()
    img = imresize(img, [224, 224])

    # run the graph
    with tf.Session(graph=graph) as sess:
        [out] = sess.run([output], feed_dict={input_maps: [img]})

        feat = {}

        print(out.keys())

        for k, v in out.items():
            if k in ['conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1',
                     'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
                     'conv5_3', 'relu5_3', 'pool5', 'fc6', 'relu6', 'fc7', 'relu7', 'fc8']:
                feat[k] = v.tolist()

        print(out)
        with open('./features/{0}.pkl'.format(img_fname.split('/')[-1].split('.')[0]), mode='wb') as f:
            pickle.dump(feat, f)

        sess.close()


if __name__ == '__main__':
    for img in os.listdir(config['scut_fbp5500_img_base_dir']):
        extract_deep_feats(os.path.join(config['scut_fbp5500_img_base_dir'], img))
