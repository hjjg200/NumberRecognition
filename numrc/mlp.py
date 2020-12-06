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

    def __init__(self, *hl_sizes, in_size=IMG_SIZE, out_size=10, \
        rand_init=True):

        if rand_init == True:
            sizes = (in_size, *hl_sizes, out_size)
            self.sizes = np.asarray(sizes)
            self.weights = [np.random.randn(j, i) \
                for i, j in zip(sizes[:-1], sizes[1:])]
            self.biases = [np.random.randn(n, 1) \
                for n in sizes[1:]]

    @classmethod
    def from_data(cls, sizes, weights, biases):
        self = cls(rand_init=False)
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
                score, _ = self.test(tdb)
                print("Epoch {0} tested: {1}% of {2}".format( \
                    i, score, len(tdb)))
            else:
                print("Epoch {0} done".format(i))

    def __run(self, batch, lr):
        grad_c_w = [np.zeros(w.shape) for w in self.weights]
        grad_c_b = [np.zeros(b.shape) for b in self.biases]
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
        self.weights = [w - (lr / len(batch)) * g \
            for w, g in zip(self.weights, grad_c_w)]
        self.biases = [b - (lr / len(batch)) * g \
            for b, g in zip(self.biases, grad_c_b)]

    def test(self, tdb, return_failed=False):

        failed = []
        for entry in tdb:
            if self.recognize(entry) != entry.label:
                failed.append(entry)

        score = round(100.0 - len(failed) / len(tdb) * 100.0, 2)
        self.last_score = score
        if return_failed:
            failed = Database.from_entries(failed)
        else:
            failed = None

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

        return cls.from_data(sizes, weights, biases)

    def evolve(self, other, epochs, offsprings_size, tdb):

        if offsprings_size < 2:
            raise Exception("Offsprings size must be above 1")
        assert np.array_equal(self.sizes, other.sizes)

        a0, a1, a2, a3 = self, self, other, other
        for i in range(epochs):
            offsprings = a0.reproduce(a2, offsprings_size)
            offsprings += a1.reproduce(a3, offsprings_size)

            offsprings.sort(key=lambda each: each.test(tdb)[0], \
                reverse=True)

            a0, a1, a2, a3 = offsprings[:4]
            print("Epoch {0} done:".format(i))
            print("- ", end="")
            print([each.last_score for each in offsprings])

        return (a0, a1)

    def reproduce(self, other, count):

        """
        Attributes
        """
        ri = np.random.randint
        attr_w = [[ri(-w.size, w.size, w.shape) for w in self.weights] \
            for _ in range(count)]
        attr_b = [[ri(-b.size, b.size, b.shape) for b in self.biases] \
            for _ in range(count)]

        # attr > 0 means it's derived from other
        # attr == 0 means it's mutated

        """
        Mutator
        """
        stdn = np.random.standard_normal
        mt_w = [[lambda a: np.mean(w) + np.std(w) * stdn(a.shape) \
            for w in self.weights] for _ in range(count)]
        mt_b = [[lambda a: np.mean(b) + np.std(b) * stdn(a.shape) \
            for b in self.biases] for _ in range(count)]

        offsprings = []
        for i in range(count):
            w = [np.copy(w) for w in self.weights]
            b = [np.copy(b) for b in self.biases]
            aw, ab = attr_w[i], attr_b[i]
            mw, mb = mt_w[i], mt_b[i]
            for j in range(len(w)):

                # Crossover
                w[j][aw[j] > 0] = other.weights[j][aw[j] > 0]
                b[j][ab[j] > 0] = other.biases[j][ab[j] > 0]

                # Mutation
                mp_w = aw[j] == 0
                mp_b = ab[j] == 0
                w[j][mp_w] = mw[j](w[j][mp_w])
                b[j][mp_b] = mb[j](b[j][mp_b])

            offsprings.append(self.__class__.from_data(self.sizes, w, b))

        return offsprings

class SigmoidMLP(MLP):
    def f(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    def f_prime(self, z):
        return self.f(z) * (1.0 - self.f(z))
    def c_prime(self, y, a):
        return a - y

class ReLUMLP(MLP):
    def g(self, z):
        return np.maximum(z, 0)
    def f(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    def f_prime(self, z):
        return self.f(z) * (1.0 - self.f(z))
    def c_prime(self, y, a):
        return a - y
    """
    def f_prime(self, z):
        r = np.copy(z)
        r[r <= 0] = 0
        r[r > 0] = 1
        return r
    """
