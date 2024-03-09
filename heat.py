import pyopencl as cl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from pathlib import Path
from functools import partial
from masks import diamond, circle

# TODO:
"""
Current thing is stupid. why do we make a whole array for each timestep?
we only need the previous one, right? (or the next one, which couldnt be?)

attempted with: u_prev(k) and u_next(k+1)
"""

# settings #
# the width of the plate
plate_width = 256
# the height of the plate
plate_height = plate_width
# how many iterations to do
max_iter_time = 750

# sim substeps
substeps = 13

# alpha?
alpha = 2
# delta_x?
delta_x = 1

# time step?
delta_t = (delta_x ** 2)/(4 * alpha)
# idk
gamma = (alpha * delta_t) / (delta_x ** 2)

# so this is an array with an array for each time step
u_curr = np.empty((plate_width, plate_height))

# initial plate tmep
u_initial = -1

# idk
u_top = 99.0
# u_left = .0
# u_bottom = .0
# u_right = .0

u_curr.fill(u_initial)

# u[:, (plate_height-1):, :] = u_top
# u[:, :, :1] = u_left
# u[:, :1, 1:] = u_bottom
# u[:, :, (plate_width-1):] = u_right

## masks
diamond_mask = diamond((0+65, plate_height-50), (plate_width, plate_height))
circle_mask = circle((plate_width / 2, plate_height / 2), (plate_width, plate_height), 50)
u_curr[diamond_mask] = u_top
u_curr[circle_mask] = 99.0

u_next = np.copy(u_curr).astype(np.float32)
# u_next = np.empty_like(u_prev, dtype=np.float32)

## opencl stuff
devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=devices)
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

program = cl.Program(ctx, Path('heatk.cl').read_text()).build()
update = partial(program.update, queue)

## funcs
def calc(in_u: np.ndarray) -> np.ndarray:
  res = np.ascontiguousarray(np.empty_like(in_u), dtype=np.float32).flatten()
  curr = np.ascontiguousarray(in_u, dtype=np.float32).flatten()
  currBuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=curr)
  resBuf = cl.Buffer(ctx, mf.WRITE_ONLY, res.nbytes)
  update(res.shape, None, currBuf, resBuf, np.int32(substeps), np.int32(plate_height), np.float32(gamma))
  cl.enqueue_copy(queue, res, resBuf)
  return res.reshape((plate_width, plate_height))

def plotheatmap(u_k: np.ndarray, k: int):
    # Clear the current plot figure
    plt.figure(1)
    plt.clf()

    plt.title(f"Temperature at t = {k*delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=-1, vmax=100)
    plt.colorbar()

    # return plt
  
def plottempdist(y: int, u_k: np.ndarray, k: int):
  # clear the current plot
  plt.figure(2)
  plt.clf()
  # plt.title(f"Temperature Distribution at t = {k*delta_t:.3f} unit time (y = {y})")
  # plt.xlabel("x")
  # plt.ylabel("Temperature")
  # plt.plot(u_k[y, :])
  plt.title(f"Sum of Temperatures Across Columns at t = {k*delta_t:.3f} unit time")
  plt.xlabel("Column Index")
  plt.ylabel("Sum of Temperatures")
    
  # Sum the values along each column
  column_sums = np.sum(u_k, axis=0)
    
  # Plot the summed values as a bar graph
  plt.bar(range(u_k.shape[1]), column_sums)

def animate(k: int):
  global u_next, u_curr

  u_curr = u_next
  u_next = calc(u_curr)
  # plt.subplot(2, 1, 1)
  # plotheatmap(u_curr, k)
  # plt.subplot(2, 1, 2)
  # plottempdist(plate_height//2, u_curr, k)

# fig = plt.figure(figsize=(10, 8))
# anim = anim.FuncAnimation(plt.figure(), animate, interval=delta_t, frames=max_iter_time-1, repeat=False)
# anim.save("cool.gif")
# plt.show()
# fig_heatmap = plt.figure(1)
# anim1 = anim.FuncAnimation(fig_heatmap, animate, interval=delta_t, frames=max_iter_time-1, repeat=False)

# temp_distr = plt.figure(2)
# anim2 = anim.FuncAnimation(temp_distr, animate, interval=delta_t, frames=max_iter_time-1, repeat=False)

# # plt.show()
# anim1.save('hatmap.gif')
# anim2.save('temp_distr.gif')
