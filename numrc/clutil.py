import pyopencl as cl
import numpy as np

plat = cl.get_platforms()[0]
dev = plat.get_devices(cl.device_type.GPU)[0]
MF = cl.mem_flags
CTX = cl.Context([dev])
Q = cl.CommandQueue(CTX, dev)

def in_buffer(a, mode='r'):
    flag = MF.COPY_HOST_PTR
    if mode == 'r':
        flag |= MF.READ_ONLY
    elif mode == 'rw':
        flag |= MF.READ_WRITE

    return cl.Buffer(CTX, flag, hostbuf=a)

