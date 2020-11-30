import numpy as np

class Printer(object):

    GRAYSCALE = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]
    TERMCOLORS = ["\x1b[48;5;%dm \x1b[0m" % n for n in \
        [231,] + list(range(255, 231, -1)) + [16,]]

    def __init__(self, shape):
        if len(shape) > 2:
            raise Exception("Shape must be 1d or 2d")
        self.shape = shape if len(shape) == 2 else (shape[0], 1)

    @staticmethod
    def float_to_char(fl):
        return Printer.GRAYSCALE[int(fl * (len(Printer.GRAYSCALE) - 1))]

    @staticmethod
    def float_to_color(fl):
        return Printer.TERMCOLORS[int(fl * (len(Printer.TERMCOLORS) - 1))]

    """
    TODO: check term color; if >= 256, print ansi colors,
    else print characters
    """
    def print(self, val, widen=2):
        copy = np.array(val).reshape(self.shape)
        for row in copy:
            buf = ""
            for col in row:
                buf += Printer.float_to_char(col) * widen
            print(buf)

    def print_color(self, val, widen=2):
        copy = np.array(val).reshape(self.shape)
        for row in copy:
            buf = ""
            for col in row:
                buf += Printer.float_to_color(col) * widen
            print(buf)

