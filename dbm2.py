import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.special import expit


class DBM(object):

    def __init__(self, n_v: int, n_hs: tuple, mini_batch: int, learning_rate: float, epoch: int,
                 gibbs_step: int, mean_field_step: int, test_gibbs_step=100000,
                 load_path=None, save_path=None):
        self.n_v = n_v
        self.n_hs = n_hs
        self.mini_batch = mini_batch
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.gibbs_step = gibbs_step
        self.mean_field_step = mean_field_step
        self.test_gibbs_step = test_gibbs_step

        try:
            (self.W1, self.W2) = pickle.load(open(load_path, "rt"))
            print("load model from ", load_path)
        except:
            self.W1 = np.random.normal(loc=0, scale=0.01, size=(n_v, n_hs[0]))
            self.W2 = np.random.normal(loc=0, scale=0.01, size=(n_hs[0], n_hs[1]))
            print("Model initialized from scratch")
        self.l_list = []
        self.train_data = None
        self.prior = None
        self.load_path = load_path
        self.save_path = save_path

    def fit(self, train_data):

        self._store_data(train_data)
        # Pre-train the first RBM
        for i in range(int(self.epoch * self.train_data.shape[0] / self.mini_batch)):
            v = self._send_batch()
            self._pre_train(v, 0)
            if i % 1000 == 0:
                print('Pre-training layer 1, step %d' % i)

        # Pre-train the second RBM
        for i in range(int(self.epoch * self.train_data.shape[0] / self.mini_batch)):
            v = self._send_batch()
            self._pre_train(v, 1)
            if i % 1000 == 0:
                print('Pre-training layer 2, step %d' % i)

        # Fine-tuning process
        v_tilde, h1_tilde, h2_tilde = self._gibbs_init()

        for i in range(int(self.epoch * self.train_data.shape[0] / self.mini_batch)):
            v = self._send_batch()
            h1, h2 = self._mean_field(v)
            v_tilde, h1_tilde, h2_tilde = self._gibbs_sampling(v_tilde, h1_tilde, h2_tilde)
            self._update_weight(v, h1, h2, v_tilde, h1_tilde, h2_tilde)
            if i % 200 == 0:
                v_test = self._send_batch()
                v_test_rec, _, _ = self.transform(v_test)
                loss = self._cross_entropy(v_test, v_test_rec)
                self.l_list.append(loss)
                print('Fine-Tuning, step %d, loss is %f' % (i, loss))

    def transform(self, v):  # TODO: Need to check out if hidden layer are binary or prob
        h1, h2 = self._mean_field(v)
        h1 = np.random.binomial(1, h1)
        h2 = np.random.binomial(1, h2)
        return self._gibbs_sampling(v, h1, h2, test=True)

    def plot_loss(self):
        plt.plot(np.arange(len(self.l_list)), self.l_list)
        plt.ylabel('Loss')
        plt.xlabel('Training step/200')
        plt.title('Entropy loss plot by training step')
        plt.show()

    def save(self, filename="dbm_model"):
        if self.save_path is None:
            save_path = './' + filename + '.pickle'
        else:
            save_path = self.save_path + '.pickle'

        pickle.dump((self.W1, self.W2), open(save_path, 'wb'))

    def _send_batch(self):
        idx = np.random.choice(range(self.train_data.shape[0]), size=self.mini_batch, replace=False)
        return self.train_data[idx]

    def _pre_train(self, v: np.ndarray, pos: int):
        if pos == 0:
            h1 = self._expit_choice(np.dot(v, self.W1) * 2)
            v_rec = expit(np.dot(h1, self.W1.T))
            self.W1 = self.W1 + self.learning_rate * (np.dot(v.T, h1) - np.dot(v_rec.T, h1)) / self.mini_batch
        else:
            h1 = self._expit_choice(np.dot(v, self.W1) * 2)
            h2 = self._expit_choice(np.dot(h1, self.W2))
            h1_rec = expit(np.dot(h2, self.W2.T) * 2)
            self.W2 = self.W2 + self.learning_rate * (np.dot(h1.T, h2) - np.dot(h1_rec.T, h2)) / self.mini_batch

    def _get_hidden_layer(self, v):  # TODO: Need to check out if hidden layer are binary or prob
        h1 = self._expit_choice(np.dot(v, self.W1) * 2)
        h2 = self._expit_choice(np.dot(h1, self.W2))
        return h1, h2

    def _mean_field(self, v):
        h1, h2 = self._get_hidden_layer(v)
        for i in range(self.mean_field_step):
            temp1 = expit(np.dot(v, self.W1) + np.dot(h2, self.W2.T))
            temp2 = expit(np.dot(h1, self.W2))
            h1, h2 = temp1, temp2
        return h1, h2

    def _gibbs_init(self):
        v_tilde = np.random.binomial(1, self.prior, size=(self.mini_batch, self.prior.shape[0]))
        h1_tilde, h2_tilde = self._get_hidden_layer(v_tilde)
        return v_tilde, h1_tilde, h2_tilde

    def _gibbs_sampling(self, v_tilde, h1_tilde, h2_tilde, test=False):
        if test:
            step = self.test_gibbs_step
        else:
            step = self.gibbs_step
        for i in range(step):
            v_tilde = expit(np.dot(h1_tilde, self.W1.T))
            h2_tilde = self._expit_choice(np.dot(h1_tilde, self.W2))
            h1_tilde = self._expit_choice(np.dot(v_tilde, self.W1) + np.dot(h2_tilde, self.W2.T))
        return v_tilde, h1_tilde, h2_tilde

    def _update_weight(self, v, h1, h2, v_tilde, h1_tilde, h2_tilde):
        self.W1 = self.W1 + self.learning_rate * (np.dot(v.T, h1) - np.dot(v_tilde.T, h1_tilde)) / self.mini_batch
        self.W2 = self.W2 + self.learning_rate * (np.dot(h1.T, h2) - np.dot(h1_tilde.T, h2_tilde)) / self.mini_batch

    def _store_data(self, train_data):
        self.train_data = train_data
        self.prior = np.mean(self.train_data, axis=0)

    @staticmethod
    def _expit_choice(x: np.ndarray):
        return (np.sign(expit(x) - np.random.uniform(size=x.shape)) + 1) / 2

    @staticmethod
    def _cross_entropy(v, v_rec):
        return - np.sum(v * np.log(v_rec) + (1-v) * np.log(1-v_rec)) / v.shape[0]


