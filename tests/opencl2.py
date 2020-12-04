import pyopencl as cl
import numpy as np
import sys

platforms = cl.get_platforms()
platform = platforms[0]
devs = platform.get_devices(cl.device_type.GPU)
dev = devs[0]
mf = cl.mem_flags
ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx, dev)

a = np.arange(24).astype(np.int32).reshape(3,4,2)
b1 = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
fmt = cl.ImageFormat(cl.channel_order.R, cl.channel_type.SIGNED_INT32)
i1 = cl.Image(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, fmt, hostbuf=a)

prog = cl.Program(ctx, """
#define GL_ID (int4)(get_global_id(1), get_global_id(0), get_global_id(2), 0)

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__kernel void Image(
    __read_only image3d_t img)
{
    int4 id = GL_ID;
    int4 cl = read_imagei(img, sampler, id);
    printf("%d, %d, %d: %d\\n", id.x, id.y, id.z, cl.x);
}

__kernel void Buffer(
    __global int *ary)
{
    int4 id = GL_ID;
    printf("%d, %d, %d, %d:\\n", id.x, id.y, id.z, \
        ary[id.z * 12 + id.y * 3 + id.x]);
}
""").build()

ev = prog.Image(queue, a.shape, None, i1)
ev.wait()
ev = prog.Buffer(queue, a.shape, None, b1)
ev.wait()
