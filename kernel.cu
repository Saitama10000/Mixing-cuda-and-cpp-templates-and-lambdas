#include "kernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define err(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename FUNC>
__global__ void f_kernel(float* a, float* b, int size, FUNC func) 
{ 
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x)
      b[i] = func(a[i]);
}

template <typename FUNC>
std::vector<float> f(std::vector<float> const& a, FUNC func)
{
    std::vector<float> b(a.size());
    const int bsize = a.size() * 4;
    float* da;
    float* db;
    err(cudaMalloc(&da, bsize));
    err(cudaMalloc(&db, bsize));
    err(cudaMemcpy(da, a.data(), bsize, cudaMemcpyHostToDevice));
    f_kernel<<<256, 256>>>(da, db, a.size(), func);
    err(cudaDeviceSynchronize());

    err(cudaMemcpy(b.data(), db, bsize, cudaMemcpyDeviceToHost));
    err(cudaFree(da));
    err(cudaFree(db));
    return b;
}

// I have to explicit instantiate but don't know how to do this for lambdas
