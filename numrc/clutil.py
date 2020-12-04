import pyopencl as cl
import numpy as np

plat = cl.get_platforms()[0]
dev = plat.get_devices(cl.device_type.GPU)[0]
MF = cl.mem_flags
CTX = cl.Context([dev])
Q = cl.CommandQueue(CTX, dev)

def in_buffer(a):
    return cl.Buffer(CTX, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=a)

