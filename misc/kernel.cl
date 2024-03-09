// we get a vector of vectors and then we need to go ahead and add each vector to a total vector
// so we get in a vec which has 2d vecs. add each 2d vecs element to a total vec

// vec is a list of size `size` of lists with 2 elements each
__kernel void vSum(__global float2* vectors, __global float* res, const int num_vecs) {
  int gid = get_global_id(0);
  float2 sum = (float2)(0.0f, 0.0f);

  for (int i = 0; i < num_vecs; ++i) {
    float2 vec = vectors[gid * num_vecs + i];
    sum.x += vec.x;
    sum.y += vec.y;
  }

  res[gid] = sum.x + sum.y;
}
