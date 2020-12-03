import numpy as np
import math
import pyopencl as cl
from enum import Enum

import clutil as clu
from .constants import IMG_ROWS, IMG_COLS, IMG_SIZE, IMG_SHAPE, \
    IMG_SQUARE

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
        self.image = image.astype(np.float32)
        self.label = label

    def print(self, widen=2):
        for row in self.image.reshape(IMG_SQUARE):
            print("".join( \
                PAINT_COLORS[int(f * (len(PAINT_COLORS) - 1))] * widen \
                for f in row))
        print("Label: %d" % self.label)

    def __replace(self, image1):
        return Entry(image1, self.label)

    """
    Cartesian - raster
    """
    @staticmethod
    def cx(rx):
        return rx - IMG_COLS / 2.0

    @staticmethod
    def cy(ry):
        return -ry + IMG_ROWS / 2.0

    @staticmethod
    def rx(cx):
        return int(cx + IMG_COLS / 2.0)

    @staticmethod
    def ry(cy):
        return int(IMG_ROWS / 2.0 - cy)

    def corner(self, x_dir, y_dir, threshold=0.0):
        """
        x_dir = 0 and y_dir > 0 will push all the pixels to rightmost
        position without losing any pixel whose value exceeds the threshold
        """
        source = self.image.reshape(IMG_SQUARE)
        projection = np.zeros(IMG_SQUARE)
        y_dir, x_dir = np.sign(y_dir), np.sign(x_dir)
        dy, dx = -1, -1
        for row, col in np.ndindex(IMG_SQUARE):
            ry0 = row if y_dir > 0 else -row - 1
            rx0 = -row - 1 if x_dir > 0 else row
            fy0 = source[ry0][col]
            fx0 = source[col][rx0]
            if dy == -1 and fy0 > threshold:
                dy = row
            if dx == -1 and fx0 > threshold:
                dx = row
        dy *= y_dir * -1 if dy >= 0 else 0
        dx *= x_dir if dx >= 0 else 0

        for ry1, rx1 in np.ndindex(IMG_SQUARE):
            ry0, rx0 = ry1 - dy, rx1 - dx
            if ry0 < 0 or ry0 >= IMG_ROWS or rx0 < 0 or rx0 >= IMG_COLS:
                continue
            projection[ry1][rx1] = source[ry0][rx0]
        return self.__replace(projection.reshape(IMG_SHAPE))

    def squeeze(self, x_factor, y_factor):
        source = self.image.reshape(IMG_SQUARE)
        projection = np.zeros(IMG_SQUARE)
        for ry1, rx1 in np.ndindex(IMG_SQUARE):
            cy1, cx1 = self.cy(ry1), self.cx(rx1)
            cy0, cx0 = cy1 / y_factor, cx1 / x_factor
            ry0, rx0 = self.ry(cy0), self.rx(cx0)
            if ry0 < 0 or ry0 >= IMG_ROWS or rx0 < 0 or rx0 >= IMG_COLS:
                continue
            projection[ry1][rx1] = source[ry0][rx0]
        return self.__replace(projection.reshape(IMG_SHAPE))

    vars_dict = {'rows': IMG_ROWS, 'cols': IMG_COLS, 'hr': IMG_ROWS / 2.0,
        'hc': IMG_COLS / 2.0}
    programs = cl.Program(clu.CTX, """
    #define ROWS %(rows)d
    #define COLS %(cols)d
    #define HR %(hr)f
    #define HC %(hc)f
    #define IDX(y, x) y * COLS + x

    __kernel void Invert(
        __global float *a,
        __global float *out) {
        int i = get_global_id(0);
        out[i] = 1.0 - a[i];
    }

    __kernel void Noise(
        __global float *a,
        __global float *d,
        __global float *out) {
        int i = get_global_id(0);
        out[i] = max(min(a[i] + d[i], 1.0), 0.0);
    }

    __kernel void Rotate(
        __global float *a,
        __global float *r,
        __global float *out) {
        int ry1 = get_global_id(0);
        int rx1 = get_global_id(1);
        float cy1 = -ry1 + HR;
        float cx1 = rx1 - HC;
        float cy0 = cx1 * r[2] + cy1 * r[3];
        int ry0 = HR - cy0;
        if(ry0 < 0 || ry0 >= ROWS) return;
        float cx0 = cx1 * r[0] + cy1 * r[1];
        int rx0 = cx0 + HC;
        if(rx0 < 0 || rx0 >= COLS) return;
        out[IDX(ry1, rx1)] = a[IDX(ry0, rx0)];
    }
    """ % vars_dict).build()

    def rotate(self, rad):
        out = np.empty_like(self.image)
        buf = clu.new_out(out)
        c, s = math.cos(rad), math.sin(rad)
        r = np.asarray([c, -s, s, c]).astype(np.float32)
        self.programs.Rotate(clu.Q, IMG_SQUARE, None, \
            clu.copy_in(self.image), clu.copy_in(r), buf)
        cl.enqueue_copy(clu.Q, out, buf)
        return self.__replace(out)

    def noise(self, factor):
        delta = (np.random.standard_normal(IMG_SHAPE) * factor) \
            .astype(np.float32)
        out = np.empty_like(self.image)
        buf = clu.new_out(out)
        self.programs.Noise(clu.Q, self.image.shape, None, \
            clu.copy_in(self.image), clu.copy_in(delta), buf)
        cl.enqueue_copy(clu.Q, out, buf)
        return self.__replace(out)

    def invert(self):
        out = np.empty_like(self.image)
        buf = clu.new_out(out)
        self.programs.Invert(clu.Q, self.image.shape, None, \
            clu.copy_in(self.image), buf)
        cl.enqueue_copy(clu.Q, out, buf)
        return self.__replace(out)


