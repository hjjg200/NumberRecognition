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
b = np.zeros(2).astype(np.int32)
b1 = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
b2 = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b)

prog = cl.Program(ctx, """
#define GL_ID (int4)(get_global_id(1), get_global_id(0), get_global_id(2), 0)

__kernel void Buffer(
    __global int *ary)
{

    int4 id = GL_ID;
    int i = id.z * 12 + id.y * 4 + id.x;
    ary[i] = ary[i] * 10;
    printf("%d %d %d: %d\\n", id.x, id.y, id.z, ary[i]);

}
""").build()

if isinstance(prog, cl.Program):
    print(True)

prog.Buffer(queue, a.shape, None, b1)
prog.Buffer(queue, a.shape, None, b1)
ev = prog.Buffer(queue, a.shape, None, b1)

ev.wait()
