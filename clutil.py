import pyopencl as cl
import numpy as np

MF = cl.mem_flags
CTX = cl.create_some_context()
Q = cl.CommandQueue(CTX)

def build(code):
    return cl.Program(CTX, code).build()

def copy_in(a):
    return cl.Buffer(CTX, MF.READ_ONLY | MF.COPY_HOST_PTR, hostbuf=a)

def copy_out(a):
    return cl.Buffer(CTX, MF.WRITE_ONLY | MF.COPY_HOST_PTR, hostbuf=a)

def new_out(a):
    return cl.Buffer(CTX, MF.WRITE_ONLY, a.nbytes)
