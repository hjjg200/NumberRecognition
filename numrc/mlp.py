import numpy as np
import abc
import time
import random
from pathlib import Path

from .constants import IMG_SIZE

class MLP(metaclass = abc.ABCMeta):

    def __init__(self, *hl_sizes, in_size=IMG_SIZE, out_size=10, rand=True):
        if rand == False:
            return
        sizes = (in_size, *hl_sizes, out_size)
        self.sizes = sizes
        self.weights = np.asarray([np.random.randn(j, i) \
            for i, j in zip(sizes[:-1], sizes[1:])])
        self.biases = np.asarray([np.random.randn(n, 1) \
            for n in sizes[1:]])

    @classmethod
    def from_data(cls, weights, biases):
        self = cls(rand=False)
        self.weights = weights
        self.biases = biases
        return self

    def recognize(self, entry):
        a = entry.image
        for w, b in zip(self.weights, self.biases):
            a = self.f(np.dot(w, a) + b)
        return np.argmax(a)

    def train(self, db, epochs, batch_size, lr, tdb=None):
        db = list(db)
        for i in range(epochs):
            random.shuffle(db)
            batches = [db[j:j + batch_size] \
                for j in range(0, len(db), batch_size)]
            for batch in batches:
                self.__run(batch, lr)
            if tdb is not None:
                self.test(tdb)
                print("Epoch {0} tested: {1}% of {2}".format( \
                    i, self.score, len(tdb)))
            else:
                print("Epoch {0} done".format(i))

    def __run(self, batch, lr):
        grad_c_w = np.asarray([np.zeros(w.shape) for w in self.weights])
        grad_c_b = np.asarray([np.zeros(b.shape) for b in self.biases])
        for entry in batch:
            a, label = entry.image, entry.label
            y = np.zeros((10, 1))
            y[label] = 1.0
            as_ = [a]
            zs = []
            for w, b in zip(self.weights, self.biases):
                z = np.dot(w, a) + b
                zs.append(z)
                a = self.f(z)
                as_.append(a)
            d_c = self.c_prime(y, as_[-1]) * self.f_prime(zs[-1])
            grad_c_w[-1] += np.dot(d_c, as_[-2].transpose())
            grad_c_b[-1] += d_c
            for j in range(2, len(self.sizes)):
                z = zs[-j]
                d_v = self.f_prime(z)
                d_c = np.dot(self.weights[-j + 1].transpose(), d_c) * d_v
                grad_c_w[-j] += np.dot(d_c, as_[-j - 1].transpose())
                grad_c_b[-j] += d_c
        self.weights -= (lr / len(batch)) * grad_c_w
        self.biases -= (lr / len(batch)) * grad_c_b

    def test(self, tdb):
        success = sum(int(self.recognize(entry) == entry.label) \
            for entry in tdb) * 1.0
        self.score = round(success / len(tdb) * 100.0, 2)
        return self.score

    @abc.abstractmethod
    def c_prime(self, y, a):
        pass

    @abc.abstractmethod
    def f(self, z):
        pass

    @abc.abstractmethod
    def f_prime(self, z):
        pass

    def save(self):
        name = "{0}_{1}".format(fake.first_name(), self.score)
        np.save(RECORDS_DIR / (name + MLP.EXT_BIASES), \
            self.biases, allow_pickle=True)
        np.save(RECORDS_DIR / (name + MLP.EXT_WEIGHTS), \
            self.weights, allow_pickle=True)
        print("Saved data to {0}.*.npy".format(name))

    def load(self, name):
        self.biases = np.load(RECORDS_DIR / (name + MLP.EXT_BIASES), \
            allow_pickle=True)
        self.weights = np.load(RECORDS_DIR / (name + MLP.EXT_WEIGHTS), \
            allow_pickle=True)
        print("Loaded data from {0}.*.npy".format(name))

    def evolve(self, other, epochs, offsprings_size, test_data):
        if offsprings_size < 2:
            raise Exception("Offsprings size must be above 1")
        a0, a1 = (self, other)
        for i in range(epochs):
            offsprings = a0.reproduce(a1, offsprings_size)
            for offspring in offsprings:
                offspring.test(test_data)
            offsprings.sort(key=lambda each: each.score, reverse=True)
            a0, a1 = offsprings[:2]
            print("Epoch {0} done:".format(i))
            print("- 1st score: {0}".format(a0.score))
            print("- 2nd score: {0}".format(a1.score))
        return (a0, a1)

    def reproduce(self, other, count):
        """ Returns array of MLP of offsprings """
        b0, b1 = (self.biases, other.biases)
        w0, w1 = (self.weights, other.weights)
        offsprings = [SigmoidMLP( \
            biases=(b0 + (b1 - b0) * 2.0 * np.random.standard_normal(size=b0.shape)), \
            weights=(w0 + (w1 - w0) * 2.0 * np.random.standard_normal(size=w0.shape))) \
            for _ in range(count)]
        for offspring in offsprings:
            offspring.__mutate()
        return offsprings

    def __mutate(self):
        """ Mutates its weights and biases """
        pass

class SigmoidMLP(MLP):
    def f(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    def f_prime(self, z):
        return self.f(z) * (1.0 - self.f(z))
    def c_prime(self, y, a):
        return a - y

"""
class ReLUMLP(MLP):
    def f(self, z):
        return np.maximum(z, 0)
    def f_prime(self, z):
        r = np.copy(z)
        r[r <= 0] = 0
        r[r > 0] = 1
        return r
    def c_prime(self, a, y):
        pass
"""

