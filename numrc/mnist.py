import numpy as np
import math

from .constants import IMG_ROWS, IMG_COLS, IMG_SIZE, IMG_SHAPE, IMG_SQUARE

"""
Utility methods
"""
def be2i(p):
    """
    MNIST database uses big endian for all integers
    """
    return int.from_bytes(p, "big")

def i2be(i, l=4):
    return i.to_bytes(l, "big")

def px2f(p):
    """
    map 0 - 255 to 0.0 - 1.0
    """
    return p / 255.0

def f2px(f):
    return i2be(int(f * 255.0), 1)

"""
Database class

Load and save operations follow the specified format in
http://yann.lecun.com/exdb/mnist/
"""
class Database(tuple):

    MAGIC_IMAGE = 2051
    MAGIC_LABEL = 2049

    def __new__(cls, self):
        return super(Database, cls).__new__(cls, self)

    @classmethod
    def load(cls, image_path, label_path):

        fm = open(image_path, 'rb', 4096)
        fl = open(label_path, 'rb', 4096)

        assert be2i(fm.read(4)) == Database.MAGIC_IMAGE
        assert be2i(fl.read(4)) == Database.MAGIC_LABEL
        # Check image count
        n = be2i(fm.read(4))
        assert n == be2i(fl.read(4))
        # Check rows and cols
        assert be2i(fm.read(8)) == (IMG_ROWS << 32) | IMG_COLS

        self = []
        for i in range(n):
            self.append(Entry(np.asarray( \
                [px2f(px) for px in fm.read(IMG_SIZE)]), \
                be2i(fl.read(1))))

        fm.close()
        fl.close()

        return cls(self)

    def save(self, image_path, label_path):

        # Open in exclusive creation mode
        fm = open(image_path, 'xb', 4096)
        fl = open(label_path, 'xb', 4096)

        n = len(self)
        fm.write(i2be(Database.MAGIC_IMAGE))
        fl.write(i2be(Database.MAGIC_LABEL))
        fm.write(i2be(n))
        fl.write(i2be(n))
        fm.write(i2be(IMG_ROWS))
        fm.write(i2be(IMG_COLS))

        for entry in self:
            fl.write(i2be(entry.label, 1))
            for f in entry.image.flatten():
                fm.write(f2px(f))

        fm.close()
        fl.close()

"""
Entry related
"""

PAINT_COLORS = tuple("\x1b[48;5;%dm \x1b[0m" % n \
    for n in range(255, 231, -1))

class Entry:

    def __init__(self, image, label):
        if image.shape != IMG_SHAPE:
            image = image.reshape(IMG_SHAPE)
        self.image = image
        self.label = label

    def print(self, widen=2):
        for row in self.image.reshape(IMG_SQUARE):
            print("".join( \
                PAINT_COLORS[int(f * (len(PAINT_COLORS) - 1))] * widen \
                for f in row))
        print("Label: %d" % self.label)

    def __replace(self, image1):
        return Entry(image1, self.label)

    def noise(self, factor):
        delta = np.random.standard_normal(IMG_SHAPE) * factor
        noised = np.maximum(0.0, np.minimum(1.0, self.image + delta))
        return self.__replace(noised)

    def invert(self):
        return self.__replace(1.0 - self.image)

    def rotate(self, rad):
        source = self.image.reshape(IMG_SQUARE)
        projection = np.zeros(IMG_SQUARE)
        hw, hh = IMG_COLS / 2.0, IMG_ROWS / 2.0
        for y1, x1 in np.ndindex(IMG_SQUARE):
            ty1 = -y1 + hh
            tx1 = x1 - hw
            ty0 = tx1 * math.sin(rad) + ty1 * math.cos(rad)
            y0 = int(-(ty0 - hh))
            if y0 < 0 or y0 >= IMG_ROWS:
                continue
            tx0 = tx1 * math.cos(rad) - ty1 * math.sin(rad)
            x0 = int(tx0 + hw)
            if x0 < 0 or x0 >= IMG_COLS:
                continue
            projection[y1][x1] = source[y0][x0]

        projection.reshape(IMG_SHAPE)
        return self.__replace(projection)

