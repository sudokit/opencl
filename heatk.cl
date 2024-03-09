#define IX(X, Y, N) Y *N + X

// TODO: calculate gamma here based on alpha, delta_x and delta_t. this would i
// guess make substeps work

kernel void update(global float *u, global float *res, const int substeps,
                   const int plate_width, const int plate_height,
                   const float alpha, const float delta_x) {
  // const int x = get_global_id(0);
  // const int y = get_global_id(1);
  // const int size = plate_width * plate_height;
  // const float delta_x = 1.0f;
  // const float delta_t = (delta_t * delta_t) / (4 * alpha);
  // const float gamma = (alpha * delta_t) / (delta_x * delta_x);
  const int size = plate_height; // idfk why height
  // res[IX(i, j, get_global_size(1))] = .5f;
  // const int size = get_global_size(1);
  // const int x = get_global_id(0);
  // const int y = get_global_id(1);
  // const int plate_size = plate_width * plate_height;
  // const int idx = (x * plate_size) + y;
  // res[idx] = .5f;
  // const idx =
  // const int x = get_global_id(0) % plate_width;
  // const int y = get_global_id(1) / plate_height;
  const int gid = get_global_id(0);

  const int x = gid % plate_height;
  const int y = gid / plate_height;
  const int idx = y * size + x;
  // res[IX(0, y + 1, size)] = x;
  // res[idx] = idx;
  // res[IX(x, y, size)] = IX(x, y + plate_height, size);
  // res[IX(x, y, size)] = IX(x, y + plate_height, size);
  // const int idx = IX(x, y, size);
  // res[idx + 1] = x;
  // res[IX(x, y, plate_height)] = x;
  // res[IX(x, y, size)] = u[IX(x + 1, y + 1, size)];
  // res[IX(x, y, size)] = u[IX(x, y, size)];
  // res[IX(x, y, plate_height)] = y; // iv got no fucking idea why height
  const float dx = delta_x / substeps;
  for (int i = 0; i < substeps; ++i) {
    if (x > 0 && x < plate_width - 1 && y > 0 && y < plate_height - 1) {
      // const float currDelta_x = dx * i;
      const float delta_t = (dx * dx) / (4 * alpha);
      const float gamma = (alpha * delta_t) / (dx * dx);
      res[idx] = gamma * (u[idx + 1] + u[idx - 1] + u[idx + plate_height] +
                          u[idx - plate_height] - 4 * u[idx]) +
                 u[idx];
      // res[IX(x, y, size)] = .1f;
      // } else {
      //   res[idx] = u[idx];
      // res[IX(x, y, size)] = .5f;
    }
  }
  // res[IX(x, y, size)]
  // res[IX(x, y)] = y;
  // res[IX(x, y)] = .5f;
  // res[IX(x, y, plate_size)] = .5f;
  // const int idx = (x * plate_size) + y;
  // res[idx] = .5f;
  // res[idx] = gamma * (u[(x+1) * plate_size + y])
  // res[IX(x, y, plate_height*plate_width)] = .5f;

  // res[IX(x, y)] = gamma * (u[IX(x + 1, y)] + u[IX(x - 1, y)] + u[IX(x, y +
  // 1)] +
  //                          u[IX(x, y - 1)] - 4 * u[IX(x, y)] + u[IX(x,
  //                          y)]);
  // if (i > 0 && i < size - 1 && j > 0 && j < size - 1) {
  // const int idx = (x * size) + y;
  // res[idx] = gamma * (u[idx+plate_size] + u[idx-plate_size] + u[idx+1] +
  // u[idx-1] - 4*u[idx] + u[idx]); res[idx] = gamma; res[idx] = 5.f;
  // res[IX(x, y, size)] = 5.f;
  // }
}