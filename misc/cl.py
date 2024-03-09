import pyopencl as cl
from functools import partial
import numpy as np
from pathlib import Path
from time import time as now
from itertools import groupby

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'


# vector element wise addition

vecs = 256*4

a = np.random.uniform(size=(vecs,vecs, 2)).astype(np.float32)
c = np.zeros((vecs, ), dtype=np.float32)

devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=devices)
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
vecBuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
resBuf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

vector_ops_prg = cl.Program(ctx, Path('kernel.cl').read_text()).build()

vSum = partial(vector_ops_prg.vSum, queue)
summ = np.empty_like(c)

# sum
opencl = now()
print(c.shape)
vSum(c.shape, None, vecBuf, resBuf, np.int32(a.shape[0]))
cl.enqueue_copy(queue, summ, resBuf)
opencl_took = now() - opencl
print("opencl took: ", opencl_took)


numpy = now()
real = [np.sum(ls) for ls in a]
numpy_took = now() - numpy
print("numpy took: ", numpy_took)
print("difference opencl vs numpy (+ = opencl took longer, - = numpy took longer): ", opencl_took-numpy_took)
