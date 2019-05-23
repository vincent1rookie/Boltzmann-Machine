"""
This script is to build a Resctricted Boltzmann Machine

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class RBM:

    def __init__(self, name: str, n_v: int, n_h: int, mini_batch: int, learning_rate: float, epoch: int,
                 load_path=None, save_path=None):
        self.name = name
        self.n_v = n_v
        self.n_h = n_h
        self.mini_batch = mini_batch
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.c_list = []

        with tf.variable_scope(self.name + '_input', reuse=tf.AUTO_REUSE):
            self.v = tf.placeholder(tf.float32, [self.mini_batch, self.n_v], name=self.name + '_v')
        with tf.variable_scope(self.name + '_params', reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable(name=self.name + '_W', shape=[self.n_v, self.n_h],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

        self.sess = tf.Session()
        self.h_prob = tf.sigmoid(tf.matmul(self.v, self.W))
        self.h = (tf.sign(self.h_prob - tf.random_uniform(shape=self.h_prob.shape, maxval=1)) + 1) / 2
        self.v_rec = tf.sigmoid(tf.matmul(self.h, tf.transpose(self.W)))
        self.W_delta = tf.matmul(tf.transpose(self.v), self.h) - tf.matmul(tf.transpose(self.v_rec), self.h)
        self.cost = tf.reduce_mean(tf.square(self.v - self.v_rec))
        self.update = [self.W.assign_add(self.learning_rate * self.W_delta / self.mini_batch),
                       self.cost]
        self.cost = tf.reduce_mean(tf.square(self.v - self.v_rec))
        self.train_data = None
        self.learning_step = 0

        # Try to save and/or reload model
        self.save_path = save_path
        self.saver = tf.train.Saver()
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)
        else:
            self.sess.run(tf.global_variables_initializer())

    def fit(self, train_data):
        self.train_data = train_data
        for i in range(int(self.epoch * self.train_data.shape[0] / self.mini_batch)):
            train_x = self._send_batch()
            self._train_step(train_x)
            self.learning_step += 1
            if self.learning_step % 20 == 0:
                print('%d step learned. Reconstruction cost for last 5 step is %f'
                      % (self.learning_step, np.mean(self.c_list[-5::1])))

    def _train_step(self, train_data: np.ndarray):
        _, cost = self.sess.run(self.update, feed_dict={self.v: train_data})
        self.c_list.append(cost)

    def plot_cost(self):
        """
        To print the restruction cost change after each training episode
        """
        plt.plot(np.arange(len(self.c_list)), self.c_list)
        plt.ylabel('Loss')
        plt.xlabel('Training Steps')
        plt.show()

    # def reconstruct(self, test_data: np.ndarray):
    #     v_hat = (tf.sign(self.v_rec - tf.random_uniform(shape=self.v_rec.shape, maxval=1)) + 1) / 2
    #     return self.sess.run(v_hat, feed_dict={self.v: test_data})

    def transform(self, test_data: np.ndarray):
        return self.sess.run((self.v_rec, self.h), feed_dict={self.v: test_data})

    def save(self):
        """
        To save the model
        """
        if self.save_path is not None:
            self.saver.save(self.sess, self.save_path)
        else:
            print("Save Path needed")

    def _send_batch(self):
        idx = np.random.choice(range(self.train_data.shape[0]), size=self.mini_batch, replace=False)
        return self.train_data[idx]


# aa = RBM('lala', 2, 2, 1, 0.01)
#
# aa.train(np.array([[1, 2]]))
