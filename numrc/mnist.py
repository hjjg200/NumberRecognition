import numpy as np
import math
import pyopencl as cl
from dataclasses import dataclass

from . import clutil as clu
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
class Database:

    MAGIC_IMAGE = 2051
    MAGIC_LABEL = 2049

    def __init__(self, images, labels):

        self.images = images
        self.labels = labels
        self.entries = [Entry(images[IMG_SIZE * i:IMG_SIZE * (i+1)], \
            labels[i]) for i in range(len(labels))]

        """
        Populate CL information
        """
        self.cl_dev = clu.Q.device
        self.cl_height = int(self.cl_dev.get_info(cl.device_info \
            .IMAGE3D_MAX_HEIGHT) / IMG_ROWS) * IMG_ROWS
        self.cl_width = int(self.cl_dev.get_info(cl.device_info \
            .IMAGE3D_MAX_WIDTH) / IMG_COLS) * IMG_COLS
        self.cl_dimension = (self.cl_height, self.cl_width)
        self.cl_size = len(self.images)
        self.cl_depth = math.ceil(self.cl_size / (self.cl_height * \
            self.cl_width))
        self.cl_per_row = int(self.cl_width / IMG_COLS)
        self.cl_per_depth = int((self.cl_width * self.cl_height) \
            / IMG_SIZE)

        """
        Build CL program
        """
        # Macros
        kernel_cl = """
#define PER_ROW {per_row:d}
#define PER_DEPTH {per_depth:d}
#define SIZE {size:d}
#define ROWS {rows:d}
#define COLS {cols:d}
#define HR {hr:f}
#define HC {hc:f}
        """.format(per_row=self.cl_per_row, per_depth=self.cl_per_depth, \
            size=IMG_SIZE, rows=IMG_ROWS, cols=IMG_COLS,\
            hr=IMG_ROWS / 2.0, hc=IMG_COLS / 2.0)

        # Rest header
        kernel_cl += """
#define CY(ry) -(ry) + HR
#define CX(rx) (rx) - HC
#define RY(cy) HR - (cy)
#define RX(cx) (cx) + HC
// ABS is for absolute coords in the image
// 1 is for destination 0 is for source
#define GL_ID_102 (int4)( \
    get_global_id(1), \
    get_global_id(0), \
    get_global_id(2), \
    0)
#define IDX(abs) abs.z * PER_DEPTH \
    + int(abs.y / ROWS) * PER_ROW \
    + int(abs.x / COLS)
#define ARY_IDX(idx, abs) idx * SIZE \
    + (abs.y % ROWS) * COLS \
    + abs.x % COLS
#define CARTESIAN(abs) (float4)( \
    (abs1.x % COLS) - HC, \
    -(abs1.y % ROWS) + HR, \
    abs.z, \
    0.0f)
#define RASTER(cart) (float4)( \
    cart.x + HC, \
    HR - cart.y, \
    cart.z, \
    0.0f)
#define NO_Z_INTERPOLATION(abs) (float4)( \
    abs.x, \
    abs.y, \
    abs.z + 0.5f, /* +0.5f makes z not interpolated */ \
    0.0f)

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t linear_sampler = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_NONE | CLK_FILTER_LINEAR;

__kernel void ArrayToImage(
    __read_only image3d_t ary,
    __write_only image3d_t img,
    const int length
)
{

    int4 ary_pos = GL_ID_102;
    int ary_idx = ary_pos.z * PER_DEPTH * SIZE + ary_pos.y * COLS \
        * PER_ROW + ary_pos.x;
    int idx = ary_idx / SIZE;

    // return if padded
    if(idx >= length) {
        return;
    }

    // Array index in current depth
    int ary_idx_in_depth = ary_idx % (PER_DEPTH * SIZE);
    // Array index in each image
    int ary_idx_in_each = ary_idx_in_depth % SIZE;

    int idx_in_depth = idx % PER_DEPTH;
    int base_y = (idx_in_depth / PER_ROW) * ROWS;
    int base_x = (idx_in_depth % PER_ROW) * COLS;

    int img_y = base_y + ary_idx_in_each / COLS;
    int img_x = base_x + ary_idx_in_each % COLS;

    int4 img_pos = (int4)(img_x, img_y, ary_pos.z, 0);

    float4 cl = read_imagef(ary, sampler, ary_pos);
    write_imagef(img, img_pos, cl);

}

__kernel void ImageToArray(
    __read_only image3d_t img,
    __write_only image3d_t ary,
    const int length
)
{

    int4 img_pos = GL_ID_102;
    int idx = img_pos.z * PER_DEPTH + int(img_pos.y / ROWS) * PER_ROW \
        + int(img_pos.x / COLS);

    if(idx >= length) return;

    int idx_in_depth = idx % PER_DEPTH;
    int ary_idx = img_pos.z * PER_DEPTH * SIZE + idx_in_depth * SIZE + \
        (img_pos.y % ROWS) * COLS + (img_pos.x % COLS);
    int ary_idx_in_depth = ary_idx % (PER_DEPTH * SIZE);
    int4 ary_pos = (int4)(
        ary_idx_in_depth % (PER_ROW * COLS),
        ary_idx_in_depth / (PER_ROW * COLS),
        img_pos.z,
        0
    );

    float4 cl = read_imagef(img, sampler, img_pos);
    write_imagef(ary, ary_pos, cl);

}
        """

        # Kernels
        kernel_cl += "\n".join([ \
            self.invert_cl, self.rotate_cl, self.noise_cl, self.scale_cl])

        self.program = cl.Program(clu.CTX, kernel_cl).build()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, key):
        return self.entries[key]

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

        images = [px2f(px) for px in fm.read(IMG_SIZE * n)]
        images = np.asarray(images, dtype=np.float32)
        labels = [l for l in fl.read(n)]
        labels = np.asarray(labels, dtype=np.int32)

        fm.close()
        fl.close()

        return cls(images, labels)

    def save(self, image_path, label_path):
        return # Disabled atm

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


    invert_cl = """
__kernel void Invert(
    __read_only image3d_t src,
    __write_only image3d_t dest,
    const int length,
    __global uint *bools
)
{

    int4 abs1 = GL_ID_102;
    int idx1 = IDX(abs1);

    // if padded return
    if(idx1 >= length) return;

    float4 cl0 = read_imagef(src, sampler, abs1);
    if(bools[idx1] > 0) {
        cl0.x = 1.0 - cl0.x;
    }
    write_imagef(dest, abs1, cl0);

}
    """

    def invert(self, bools):
        bools = clu.in_buffer(bools.astype(np.uint32))
        self.__run_program(self.program.Invert, bools)

    rotate_cl = """
__kernel void Rotate(
    __read_only image3d_t src,
    __write_only image3d_t dest,
    const int length,
    __global float *rads
)
{

    int4 abs1 = GL_ID_102;
    int idx1 = IDX(abs1);

    // if padded return
    if(idx1 >= length) return;

    // Cartesian
    float4 cart1 = CARTESIAN(abs1);
    float4 rast1 = RASTER(cart1);

    // Rotate
    float4 cart0 = cart1;
    float c = cos(rads[idx1]);
    float s = sin(rads[idx1]);
    cart0.y = cart1.x * s + cart1.y * c;
    cart0.x = cart1.x * c - cart1.y * s;
    float4 rast0 = RASTER(cart0);

    // Get source color
    float4 cl0 = (float4)(0.0f, 0.0f, 0.0f, 1.0f);

    if(rast0.x >= 0.0f && rast0.x < COLS &&
       rast0.y >= 0.0f && rast0.y < ROWS)
    {
        float4 abs0 = convert_float4(abs1) + (rast0 - rast1);
        abs0 = NO_Z_INTERPOLATION(abs0);
        cl0 = read_imagef(src, linear_sampler, abs0);
    }

    write_imagef(dest, abs1, cl0);

}
    """
    def rotate(self, rads):
        rads = clu.in_buffer(rads.astype(np.float32))
        self.__run_program(self.program.Rotate, rads)

    noise_cl = """
__kernel void Noise(
    __read_only image3d_t src,
    __write_only image3d_t dest,
    const int length,
    __global float *delta)
{

    int4 abs1 = GL_ID_102;
    int idx1 = IDX(abs1);

    if(idx1 >= length) return;

    int ary_idx1 = ARY_IDX(idx1, abs1);

    float4 cl0 = read_imagef(src, sampler, abs1);
    cl0.x = max(min(cl0.x + delta[ary_idx1], 1.0f), 0.0f);
    write_imagef(dest, abs1, cl0);

}
    """
    def noise(self, factor):
        delta = clu.in_buffer(((np.random.random_sample( \
            len(self.images)) * 2.0 - 1.0) * factor).astype(np.float32))
        self.__run_program(self.program.Noise, delta)

    scale_cl = """
__kernel void Scale(
    __read_only image3d_t src,
    __write_only image3d_t dest,
    const int length,
    __global float *x_factors,
    __global float *y_factors)
{

    int4 abs1 = GL_ID_102;
    int idx1 = IDX(abs1);

    if(idx1 >= length) return;

    // Cartesian
    float4 cart1 = CARTESIAN(abs1);
    float4 rast1 = RASTER(cart1);

    // Hadamard product
    float4 vec = (float4)(
        1.0f / x_factors[idx1],
        1.0f / y_factors[idx1],
        1.0f,
        1.0f);

    float4 cart0 = cart1 * vec;
    float4 rast0 = RASTER(cart0);
    float4 abs0 = convert_float4(abs1) + (rast0 - rast1);
    abs0 = NO_Z_INTERPOLATION(abs0);

    // Determine the color
    float4 cl0 = (float4)(0.0f, 0.0f, 0.0f, 1.0f);

    // Find out if it is out of the bounds
    if(rast0.x >= 0.0f && rast0.x < COLS &&
       rast0.y >= 0.0f && rast0.y < ROWS)
    {
        cl0 = read_imagef(src, linear_sampler, abs0);
    }

    write_imagef(dest, abs1, cl0);

}
    """
    def scale(self, x_factors, y_factors):
        x_factors = clu.in_buffer(x_factors.astype(np.float32))
        y_factors = clu.in_buffer(y_factors.astype(np.float32))
        self.__run_program(self.program.Scale, x_factors, y_factors)

    """
    OpenCL run program
    """
    def __run_program(self, fun, *args):

        # Length is count of complete images
        length = np.int32(len(self.entries))

        """
        Set dimensions and depth
        """
        height = self.cl_height
        width = self.cl_width
        dim = self.cl_dimension
        size = self.cl_size
        depth = self.cl_depth

        """
        Pad empty pixels
        """
        pad_size = depth * (height * width) - size
        padded = np.append(self.images, np.zeros(pad_size)) \
            .astype(np.float32)

        """
        Channel Order R and Channel Type Float gives one channeled float
        type of image
        """
        fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT)
        region = (*dim, depth)

        """
        Two image3d_t are used for image processing
        """
        img_A = cl.Image(clu.CTX, clu.MF.READ_WRITE | clu.MF.COPY_HOST_PTR,\
            fmt, shape=region, hostbuf=padded)
        img_B = cl.Image(clu.CTX, clu.MF.READ_WRITE, fmt, shape=region)
        self.program.ArrayToImage(clu.Q, region, None, \
            img_A, img_B, length)

        """
        Run the function
        """
        fun(clu.Q, region, None, img_B, img_A, length, *args)
        self.program.ImageToArray(clu.Q, region, None, \
            img_A, img_B, length)
        cl.enqueue_copy(clu.Q, padded, img_B, origin=(0,0,0), \
            region=region)

        np.copyto(self.images, padded[:len(self.images)])

"""
Entry related
"""

PAINT_COLORS = tuple("\x1b[48;5;%dm \x1b[0m" % n \
    for n in range(255, 231, -1))

class Entry:

    def __init__(self, image, label):
        self.image = np.asarray(image, dtype=np.float32).reshape(IMG_SHAPE)
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
"""
    def corner(self, x_dir, y_dir, threshold=0.0):
        " ""
        x_dir = 0 and y_dir > 0 will push all the pixels to rightmost
        position without losing any pixel whose value exceeds the threshold
        " ""
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
        out = np.empty_like(self.image)
        buf = clu.new_out(out)
        self.programs.Squeeze(clu.Q, IMG_SQUARE, None, \
            clu.copy_in(self.image), np.float32(y_factor), \
            np.float32(x_factor), buf)
        cl.enqueue_copy(clu.Q, out, buf)
        return self.__replace(out)
"""
