import numpy as np
import math

class Augmenter(object):

    def __init__(self, shape):
        self.shape = shape

    def reshape(self, val):
        return np.array(val).reshape(self.shape)

    def noise(self, val, deg):
        copy = self.reshape(val)
        delta = np.random.standard_normal(self.shape) * deg
        return np.minimum(1.0, np.maximum(0.0, copy + delta))

    def invert(self, val):
        return 1.0 - np.array(val)

    def rotate(self, val, rad):
        " rad to clockwise "
        rad = 2 * math.pi - rad
        copy = self.reshape(val)
        projection = np.zeros(self.shape) - 1.0
        hw = self.shape[1] / 2.0
        hh = self.shape[0] / 2.0
        for y, row in enumerate(copy):
            ty = -y + hh
            for x, _ in enumerate(row):
                tx = x - hw
                py = tx * math.sin(rad) + ty * math.cos(rad)
                py = -(py - hh)
                py = max(min(int(py), self.shape[0] - 1), 0)
                px = tx * math.cos(rad) - ty * math.sin(rad)
                px = px + hw
                px = max(min(int(px), self.shape[1] - 1), 0)
                projection[py][px] = copy[y][x]
        return projection
