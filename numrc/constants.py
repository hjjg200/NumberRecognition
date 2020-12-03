import pyopencl as cl

CL_CTX = cl.create_some_context()
CL_Q = cl.CommandQueue(CL_CTX)

IMG_ROWS = 28
IMG_COLS = 28
IMG_SIZE = IMG_ROWS * IMG_COLS
IMG_SHAPE = (IMG_SIZE, 1)
IMG_SQUARE = (IMG_ROWS, IMG_COLS)
