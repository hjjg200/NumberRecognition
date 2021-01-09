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

        self.images = np.asarray(images, dtype=np.float32).flatten()
        self.labels = np.asarray(labels, dtype=np.int32).flatten()
        self.entries = [Entry(self.images[IMG_SIZE * i:IMG_SIZE * (i+1)], \
            self.labels[i]) for i in range(len(self.labels))]

        """
        Populate CL information
        """
        self.cl_dev = clu.Q.device
        self.cl_height = int(self.cl_dev.get_info(cl.device_info \
            .IMAGE3D_MAX_HEIGHT) / IMG_ROWS) * IMG_ROWS
        self.cl_width = int(self.cl_dev.get_info(cl.device_info \
            .IMAGE3D_MAX_WIDTH) / IMG_COLS) * IMG_COLS
        self.cl_size = len(self.images)
        self.cl_depth = math.ceil(self.cl_size / (self.cl_height * \
            self.cl_width))
        self.cl_region = (self.cl_height, self.cl_width, self.cl_depth)
        self.cl_per_row = int(self.cl_width / IMG_COLS)
        self.cl_per_depth = int((self.cl_width * self.cl_height) \
            / IMG_SIZE)
        self.cl_format = cl.ImageFormat(cl.channel_order.R, \
            cl.channel_type.FLOAT)
        self.cl_length = np.int32(len(self.entries))

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
            hr=(IMG_ROWS - 1) / 2.0, hc=(IMG_COLS - 1) / 2.0)

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
    (abs.x % COLS) - HC, \
    -(abs.y % ROWS) + HR, \
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
#define SIGN_INT(i) ((i > 0) - (i < 0))

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
            self.invert_cl, self.rotate_cl, self.noise_cl, self.scale_cl, \
            self.corner_cl])

        self.program = cl.Program(clu.CTX, kernel_cl).build()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, key):
        return self.entries[key]

    def clone(self):
        return self.__class__(np.copy(self.images), np.copy(self.labels))

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
        labels = [l for l in fl.read(n)]

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

    @classmethod
    def from_entries(cls, entries):
        images = []
        labels = []
        for e in entries:
            images += [*e.image]
            labels += [e.label]
        return cls(images, labels)

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

        bools = self.__to_npary(bools, np.uint32)
        bools = clu.in_buffer(bools)

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

        rads = self.__to_npary(rads, np.float32)
        rads = clu.in_buffer(rads)

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
    def noise(self, factors):

        factors = self.__to_npary(factors, np.float32)
        factors = np.repeat(factors, IMG_SIZE)

        delta = clu.in_buffer(((np.random.random_sample(self.cl_size) * \
            2.0 - 1.0) * factors).astype(np.float32))

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

    // Determine the color
    float4 cl0 = (float4)(0.0f, 0.0f, 0.0f, 1.0f);

    // Find out if it is out of the bounds
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
    def scale(self, x_factors, y_factors):

        x_factors = self.__to_npary(x_factors, np.float32)
        x_factors = clu.in_buffer(x_factors)
        y_factors = self.__to_npary(y_factors, np.float32)
        y_factors = clu.in_buffer(y_factors)

        self.__run_program(self.program.Scale, x_factors, y_factors)

    corner_cl = """
__kernel void Corner1(
    __read_only image3d_t src,
    __write_only image3d_t dest,
    const int length,
    __global int *x_dirs,
    __global int *y_dirs,
    const float threshold)
{

    int4 abs1 = GL_ID_102;
    int idx1 = IDX(abs1);

    if(idx1 >= length) return;

    if(abs1.x % COLS == 0 && abs1.y % ROWS == 0)
    {
        x_dirs[idx1] = SIGN_INT(x_dirs[idx1]) * COLS;
        y_dirs[idx1] = SIGN_INT(y_dirs[idx1]) * ROWS;
    }

}

__kernel void Corner2(
    __read_only image3d_t src,
    __write_only image3d_t dest,
    const int length,
    __global int *x_dirs,
    __global int *y_dirs,
    const float threshold)
{

    int4 abs1 = GL_ID_102;
    int idx1 = IDX(abs1);

    if(idx1 >= length) return;

    float4 cart1 = CARTESIAN(abs1);

    int x_sign = SIGN_INT(x_dirs[idx1]);
    int y_sign = SIGN_INT(y_dirs[idx1]);

    // Find source pixel and its color
    float4 cl0 = read_imagef(src, sampler, abs1);
    if(cl0.x > threshold)
    {
        if(x_sign > 0)
        {
            int val = int(HC - cart1.x) + 1;
            atomic_min(&x_dirs[idx1], val);
        }
        else if(x_sign < 0)
        {
            int val = -int(cart1.x + HC) - 1;
            atomic_max(&x_dirs[idx1], val);
        }

        if(y_sign > 0)
        {
            int val = int(HR - cart1.y) + 1;
            atomic_min(&y_dirs[idx1], val);
        }
        else if(y_sign < 0)
        {
            int val = -int(cart1.y + HR) - 1;
            atomic_max(&y_dirs[idx1], val);
        }
    }

}

__kernel void Corner3(
    __read_only image3d_t src,
    __write_only image3d_t dest,
    const int length,
    __global int *x_dirs,
    __global int *y_dirs,
    const float threshold)
{

    int4 abs1 = GL_ID_102;
    int idx1 = IDX(abs1);

    if(idx1 >= length) return;

    //
    int x_sign = SIGN_INT(x_dirs[idx1]);
    int y_sign = SIGN_INT(y_dirs[idx1]);

    int dx = x_dirs[idx1] - x_sign;
    int dy = y_dirs[idx1] - y_sign;

    // Coords
    float4 cart0 = CARTESIAN(abs1);
    float4 rast1 = RASTER(cart0);

    // Update source coords
    cart0.x -= float(dx);
    cart0.y -= float(dy);
    float4 rast0 = RASTER(cart0);

    float4 cl0 = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    if(rast0.x >= 0.0f && rast0.x < COLS &&
       rast0.y >= 0.0f && rast0.y < ROWS)
    {
        float4 abs0 = convert_float4(abs1) + (rast0 - rast1);
        cl0 = read_imagef(src, sampler, abs0);
    }

    write_imagef(dest, abs1, cl0);

}
    """
    def corner(self, x_dirs, y_dirs, threshold=0.0):

        x_dirs = self.__to_npary(x_dirs, np.int32)
        x_dirs = clu.in_buffer(x_dirs, 'rw')
        y_dirs = self.__to_npary(y_dirs, np.int32)
        y_dirs = clu.in_buffer(y_dirs, 'rw')
        threshold = np.float32(threshold)

        pr = self.program
        self.__run_program( \
            [pr.Corner1, pr.Corner2, pr.Corner3], \
            x_dirs, y_dirs, threshold)

    """
    NumPy helper
    """
    def __to_npary(self, val, dtype, length=None):

        if length is None:
            length = len(self.entries)

        npary = None
        if isinstance(val, np.ndarray):
            npary = val.astype(dtype)
        elif isinstance(val, (list, tuple)):
            npary = np.array(val, dtype=dtype)
        else:
            return np.array([val] * length, dtype=dtype)

        assert len(npary) == length
        return npary

    """
    OpenCL run program
    """
    def start_filters(self):

        length = self.cl_length
        fmt = self.cl_format
        height = self.cl_height
        width = self.cl_width
        size = self.cl_size
        depth = self.cl_depth
        region = self.cl_region

        """
        Pad empty pixels
        """
        pad_size = depth * (height * width) - size
        self.cl_padded = np.append(self.images, np.zeros(pad_size)) \
            .astype(np.float32)

        """
        Two image3d_t are used for image processing
        """
        img_a = cl.Image(clu.CTX, clu.MF.READ_WRITE | clu.MF.COPY_HOST_PTR,\
            fmt, shape=region, hostbuf=self.cl_padded)
        img_b = cl.Image(clu.CTX, clu.MF.READ_WRITE, fmt, shape=region)
        self.program.ArrayToImage(clu.Q, region, None, \
            img_a, img_b, length)

        self.cl_src = img_b
        self.cl_dest = img_a

    def __run_program(self, func, *args):

        """
        If func is a list of functions, list's functions
        are run sequentially without the order of source and destination
        being changed
        """
        assert self.cl_src is not None
        assert self.cl_dest is not None
        assert self.cl_padded is not None

        """
        Set dimensions and depth
        """
        length = self.cl_length
        region = self.cl_region

        """
        Run the function
        """
        if isinstance(func, list):
            for subfunc in func:
                subfunc(clu.Q, region, None, self.cl_src, self.cl_dest, \
                    length, *args)
        else:
            func(clu.Q, region, None, self.cl_src, self.cl_dest, length, \
                *args)

        self.cl_src, self.cl_dest = self.cl_dest, self.cl_src

    def flush_filters(self):

        length = self.cl_length
        region = self.cl_region
        padded = self.cl_padded

        self.program.ImageToArray(clu.Q, region, None, \
            self.cl_src, self.cl_dest, length)
        cl.enqueue_copy(clu.Q, padded, self.cl_dest, \
            origin=(0,0,0), region=region)

        np.copyto(self.images, padded[:len(self.images)])

        self.cl_padded = None
        self.cl_src = None
        self.cl_dest = None


"""
Entry related
"""

PAINT_COLORS = tuple("\x1b[48;5;%dm \x1b[0m" % n \
    for n in range(255, 231, -1))

class Entry:

    def __init__(self, image, label):
        self.image = image.reshape(IMG_SHAPE)
        self.label = label

    def print(self, widen=2):
        for row in self.image.reshape(IMG_SQUARE):
            print("".join( \
                PAINT_COLORS[int(f * (len(PAINT_COLORS) - 1))] * widen \
                for f in row))
        print("Label: %d" % self.label)

