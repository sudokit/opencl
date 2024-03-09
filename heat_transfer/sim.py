import numpy as np
import pyopencl as cl
from pathlib import Path
from functools import partial

from dataclasses import dataclass

@dataclass
class Size:
  width: int
  height: int

class Sim:
  def __init__(self, plateDims: Size, alpha: float, delta_x: float, plateInitialTemp: int) -> None:
    self.dims = plateDims
    # self.maxIters = maxIterTime
    self.alpha = alpha
    self.delta_x = delta_x
    self.palteInitialTemp = plateInitialTemp
    
    # self.delta_t = (delta_x**2)/(4*alpha)
    # self.gamma = (alpha * self.delta_t) / (delta_x ** 2)
    
    self.u_curr = np.empty((plateDims.width, plateDims.height))
    self.u_curr.fill(plateInitialTemp)
    self.u_next = np.copy(self.u_curr).astype(np.float32)

    self.devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
    self.ctx = cl.Context(devices=self.devices)
    self.queue = cl.CommandQueue(self.ctx)
    
    self.mf = cl.mem_flags
    self.program = cl.Program(self.ctx, Path('heatk.cl').read_text()).build()
    self.update = partial(self.program.update, self.queue)
  
  def addMask(self, mask: np.ndarray, temp: float):
    self.u_curr[mask] = temp;
    self.u_next[mask] = temp;
    
  def calc(self, in_u: np.ndarray, substeps: int, gpu: bool = True) -> np.ndarray:
    assert substeps >= 2, f"Argument 'substeps' should be over 2. Got {substeps}"
    if gpu:
      res = np.ascontiguousarray(np.empty_like(in_u), dtype=np.float32).flatten()
      curr = np.ascontiguousarray(in_u, dtype=np.float32).flatten()
      currBuf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=curr)
      resBuf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, res.nbytes)
      # self.update(res.shape, None, currBuf, resBuf, np.int32(substeps), np.int32(self.dims.width), np.int32(self.dims.height), np.float32(self.alpha), np.float32(self.delta_x))
      self.update(res.shape, None, currBuf, resBuf, np.int32(substeps), np.int32(self.dims.width), np.int32(self.dims.height), np.float32(self.alpha), np.float32(self.delta_x))
      cl.enqueue_copy(self.queue, res, resBuf)
      return res.reshape((self.dims.width, self.dims.height))
    # cpu
    else :
      res = np.empty_like(in_u, dtype=np.float32)
      width, height = res.shape
      for i in range(1, width-1):
        for j in range(1, height-1):
          res[i, j] = self.gamma * (in_u[i+1][j] + in_u[i-1][j] + in_u[i][j+1] + in_u[i][j-1] - 4*in_u[i][j]) + in_u[i][j]
      return res
