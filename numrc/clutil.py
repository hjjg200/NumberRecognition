import pyopencl as cl
import numpy as np

plat = cl.get_platforms()[0]
dev = plat.get_devices(cl.device_type.GPU)[0]
MF = cl.mem_flags
CTX = cl.Context([dev])
Q = cl.CommandQueue(CTX, dev)

def build(code):
    return cl.Program(CTX, code).build()

def copy_in(a):
    return cl.Buffer(CTX, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=a)

def copy_out(a):
    return cl.Buffer(CTX, MF.WRITE_ONLY | MF.COPY_HOST_PTR, hostbuf=a)

def new_out(a):
    return cl.Buffer(CTX, MF.WRITE_ONLY, a.nbytes)
