import numpy as np
import abc
import time
import random
from pathlib import Path
import tarfile as tf
from io import BytesIO

from .constants import IMG_SIZE
from .mnist import Database

class MLP(metaclass = abc.ABCMeta):

    def __init__(self, *hl_sizes, in_size=IMG_SIZE, out_size=10, rand=True):
        if rand == False:
            return
        sizes = (in_size, *hl_sizes, out_size)
        self.sizes = np.asarray(sizes)
        self.weights = np.asarray([np.random.randn(j, i) \
            for i, j in zip(sizes[:-1], sizes[1:])])
        self.biases = np.asarray([np.random.randn(n, 1) \
            for n in sizes[1:]])

    @classmethod
    def from_data(cls, sizes, weights, biases):
        self = cls(rand=False)
        self.sizes = sizes
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
                score, failed = self.test(tdb)
                self.__run(failed, lr)
                print("Epoch {0} tested: {1}% of {2}".format( \
                    i, score, len(tdb)))
            else:
                print("Epoch {0} done".format(i))

    def __run(self, batch, lr):
        grad_c_w = np.asarray([np.zeros(w.shape) for w in self.weights])
        grad_c_b = np.asarray([np.zeros(b.shape) for b in self.biases])
        for entry in batch:
            a = entry.image
            y = np.zeros((self.sizes[-1], 1))
            y[entry.label] = 1.0
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
        failed = []
        for entry in tdb:
            if self.recognize(entry) != entry.label:
                failed.append(entry)
        failed = Database(failed)
        score = round(100.0 - len(failed) / len(tdb) * 100.0, 2)
        return score, failed

    @abc.abstractmethod
    def c_prime(self, y, a):
        pass

    @abc.abstractmethod
    def f(self, z):
        pass

    @abc.abstractmethod
    def f_prime(self, z):
        pass

    """
    Uses .npx as it does not follow either .npy or .npz format
    """
    FNAME_SHAPES = "shapes.npx"
    FNAME_SIZES = "sizes.npx"
    FNAME_WEIGHTS = "weights{0}.npx"
    FNAME_BIASES = "biases{0}.npx"

    def save(self, path):
        path = Path(path)
        tar = tf.open(path, 'x:gz', format=tf.PAX_FORMAT)

        def put(name, arr, dtype=np.float64):
            nonlocal tar
            ti = tf.TarInfo(name)
            buf = np.array(arr, dtype).tobytes()
            ti.size = len(buf)
            tar.addfile(ti, BytesIO(buf))

        put(MLP.FNAME_SIZES, self.sizes, np.int32)
        shapes = []

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            s_w, s_b = w.shape, b.shape
            shapes += [len(s_w), *s_w, len(s_b), *s_b]
            put(MLP.FNAME_WEIGHTS.format(i), w)
            put(MLP.FNAME_BIASES.format(i), b)

        put(MLP.FNAME_SHAPES, np.asarray(shapes), np.int32)

        tar.close()

    @classmethod
    def load(cls, path):
        path = Path(path)
        tar = tf.open(path, 'r:gz')

        def get(name, dtype=np.float64):
            nonlocal tar
            ti = tar.getmember(name)
            rd = tar.extractfile(ti)
            return np.frombuffer(rd.read(ti.size), dtype)

        sizes = get(cls.FNAME_SIZES, np.int32)
        shapes = get(cls.FNAME_SHAPES, np.int32)
        def next_shape():
            nonlocal shapes
            l = shapes[0]
            shape = shapes[1:1+l]
            shapes = shapes[1+l:]
            return shape

        weights, biases = [], []
        for i in range(len(sizes) - 1):
            weights.append(get(cls.FNAME_WEIGHTS.format(i)) \
                .reshape(next_shape()))
            biases.append(get(cls.FNAME_BIASES.format(i)) \
                .reshape(next_shape()))

        weights = np.asarray(weights)
        biases = np.asarray(biases)

        return cls.from_data(sizes, weights, biases)

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

