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
b_out = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)

prog = cl.Program(ctx, """
__kernel void prog(
    int num,
    __global int* n1,
    __global int *n2,
    __global int *out,
    __local int *locals)
{
    printf("%d\\n", num);
    printf("%d %d", locals[0], locals[1]);
    int i = get_global_id(0);
    out[i] = n1[i] + n2[i];
}
""").build()

prog.prog(queue, (10,), (2,), b_n1, b_n2, b_out)
cl.enqueue_copy(queue, out, b_out)

print(out)
