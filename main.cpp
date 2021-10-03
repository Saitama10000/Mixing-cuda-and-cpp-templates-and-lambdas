#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <cmath>
#include "kernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


int main()
{
   auto func = [] __host__ __device__ (float x) { return x * x; }; // random computation

   constexpr const int size = 1 << 10;
   std::vector<float> a(size);
   float pi = 4 * atan(1);
   float dx = pi / 2.0f / (size - 1);
   for (int i = 0; i < size; i++)
      a[i] = func(i * dx) * dx;      

   auto b = f(a, func);
   float sum_a = std::accumulate(a.begin(), a.end(), 0.0f);
   float sum_b = std::accumulate(b.begin(), b.end(), 0.0f);
   std::cout << "sum of a = " << sum_a << "\n";
   std::cout << "sum of b = " << sum_b << "\n";
}
