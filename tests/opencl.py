import pyopencl as cl
import numpy as np
import sys

mf = cl.mem_flags
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n1 = np.arange(10).astype(np.int32)
n2 = np.arange(10).astype(np.int32)
out = np.zeros(10).astype(np.int32)

b_n1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=n1)
b_n2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=n2)
b_out = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=out)

prog = cl.Program(ctx, """
__kernel void prog(
    __global int* n1,
    __global int *n2,
    __global int *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    printf("%d %d\\n", i, j);
    out[i] = n1[i] + n2[i];
    printf("%d ", i);
}
""").build()

ev = prog.prog(queue, (5,2), None, b_n1, b_n2, b_out)
ev.wait()

print(out)
