from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal

sys.path.append('../')
from util.file_util import prepare_scutfbp5500

plt.style.use('ggplot')


def build_toy_dataset(N, w):
    """
    for toy experiments
    :param N:
    :param w:
    :return:
    """
    D = len(w)
    x = np.random.normal(0.0, 2.0, size=(N, D))
    y = np.dot(x, w) + np.random.normal(0.0, 0.01, size=N)

    return x, y


def main():
    X_train, y_train, X_test, y_test, train_filenames, test_filenames = prepare_scutfbp5500(
        feat_layers=["conv4_1", "conv5_1"])

    N = 3300
    D = len(X_train[0])

    X = tf.placeholder(tf.float32, [N, D])
    w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
    b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
    y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N))

    qw = Normal(loc=tf.get_variable("qw/loc", [D]),
                scale=tf.nn.softplus(tf.get_variable("qw/scale", [D])))
    qb = Normal(loc=tf.get_variable("qb/loc", [1]),
                scale=tf.nn.softplus(tf.get_variable("qb/scale", [1])))

    inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
    inference.run(n_samples=3300, n_iter=250)

    y_post = ed.copy(y, {w: qw, b: qb})

    print("Mean squared error on test data:")
    print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))

    print("Mean absolute error on test data:")
    print(ed.evaluate('mean_absolute_error', data={X: X_test, y_post: y_test}))


def visualise(X_data, y_data, w, b, n_samples=10):
    """
    visualize data points
    :param X_data:
    :param y_data:
    :param w:
    :param b:
    :param n_samples:
    :return:
    """
    w_samples = w.sample(n_samples)[:, 0].eval()
    b_samples = b.sample(n_samples).eval()
    plt.scatter(X_data[:, 0], y_data)
    plt.ylim([-10, 10])
    inputs = np.linspace(-8, 8, num=400)
    for ns in range(n_samples):
        output = inputs * w_samples[ns] + b_samples[ns]
        plt.plot(inputs, output)


if __name__ == "__main__":
    main()
