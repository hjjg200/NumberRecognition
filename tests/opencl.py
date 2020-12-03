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

n1 = np.arange(10).astype(np.int32)
n2 = np.arange(10).astype(np.int32)
out = np.zeros(10).astype(np.int32)

b_n1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=n1)
b_n2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=n2)
b_out = cl.Buffer(ctx, mf.WRITE_ONLY, size=out.nbytes)

prog = cl.Program(ctx, """
__kernel void prog(
    __global int *n1,
    __global int *n2,
    __global int *out)
{
    int i = get_local_id(0);
    __local int a;
    a = i;
    barrier(CLK_LOCAL_MEM_FENCE);
    printf("%d:%d\\n", get_global_id(0), get_group_id(1));
}
""").build()

ev = prog.prog(queue, (10,1), (2,1), b_n1, b_n2, b_out)
#cl.enqueue_copy(queue, out, b_out)
ev.wait()
print(out)
