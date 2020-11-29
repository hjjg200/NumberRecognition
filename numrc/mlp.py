import numpy as np
import abc
import time
import sys
import random
from pathlib import Path
from .mnist import load_data_wrapper
from faker import Faker

Faker.seed(time.time())
fake = Faker()

PROJECT_DIR = Path(__file__).parent if __file__ else Path.cwd()
RECORDS_DIR = PROJECT_DIR / "records"
RECORDS_DIR.mkdir(exist_ok=True)

class MLP(metaclass = abc.ABCMeta):

    EXT_BIASES = ".biases.npy"
    EXT_WEIGHTS = ".weights.npy"

    def __init__(self, sizes=None, weights=None, biases=None):
        if sizes is not None:
            self.sizes = sizes
            self.biases = np.asarray([np.random.randn(n, 1) \
                for n in sizes[1:]])
            self.weights = np.asarray([np.random.randn(j, i) \
                for i, j in zip(sizes[:-1], sizes[1:])])
        elif weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            """ Apply weights and biases respectively """
            a = self.f(np.dot(w, a) + b)
        return a

    def train(self, data, epochs, batch_size, lr, test_data=None):
        for i in range(epochs):
            random.shuffle(data)
            batches = [data[j:j + batch_size] \
                for j in range(0, len(data), batch_size)]
            for batch in batches:
                self.__run(batch, lr)
            if test_data is not None:
                self.test(test_data)
                print("Epoch {0} tested: {1}/{2}".format( \
                    i, self.score, len(test_data)))
            else:
                print("Epoch {0} done".format(i))

    def __run(self, batch, lr):
        grad_c_b = np.asarray([np.zeros(b.shape) for b in self.biases])
        grad_c_w = np.asarray([np.zeros(w.shape) for w in self.weights])
        for i, y in batch: # input and desired value
            a = i
            activations = [a]
            zs = []
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, a) + b
                zs.append(z)
                a = self.f(z)
                activations.append(a)
            #
            d_c = self.c_prime(activations[-1], y) * \
                self.f_prime(zs[-1])
            grad_c_b[-1] += d_c
            grad_c_w[-1] += np.dot(d_c, activations[-2].transpose())
            #
            for j in range(2, len(self.sizes)):
                z = zs[-j]
                d_v = self.f_prime(z)
                d_c = np.dot(self.weights[-j + 1].transpose(), d_c) * d_v
                grad_c_b[-j] += d_c
                grad_c_w[-j] += np.dot(d_c, activations[-j - 1].transpose())
        #
        self.biases -= (lr / len(batch)) * grad_c_b
        self.weights -= (lr / len(batch)) * grad_c_w

    def test(self, data):
        results = [(np.argmax(self.feedforward(i)), y) \
            for i, y in data]
        self.score = sum(int(i == y) for (i, y) in results)
        return self.score

    @abc.abstractmethod
    def c_prime(self, a, y):
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
    def c_prime(self, a, y):
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

if __name__ == "__main__" and False:
    # Check for args
    mlp = SigmoidMLP([784,30,10])
    if(len(sys.argv) > 1):
        mlp.load(sys.argv[1])
    mlp.train(data, 30, 10, 3.0, test_data)
    mlp.save()

