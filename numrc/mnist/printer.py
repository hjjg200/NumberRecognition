import numpy as np

class Printer(object):

    GRAYSCALE = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

    def __init__(self, shape):
        if len(shape) > 2:
            raise Exception("Shape must be 1d or 2d")
        self.shape = shape if len(shape) == 2 else (shape[0], 1)

    @staticmethod
    def float_to_char(fl):
        return Printer.GRAYSCALE[int(fl * (len(Printer.GRAYSCALE) - 1))]

    def print(self, val):
        copy = np.array(val).reshape(self.shape)
        for row in copy:
            buf = ""
            for col in row:
                buf += Printer.float_to_char(col)
            print(buf)
