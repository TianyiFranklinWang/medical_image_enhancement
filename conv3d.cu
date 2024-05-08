#include <iostream>
#include <cstdint>
#include "conv3d.cuh"

template __global__ void convolution3d<float>(float *, float *, float *, dim3, dim3, dim3);

template __global__ void convolution3d<double>(double *, double *, double *, dim3, dim3, dim3);

template __global__ void convolution3d<int8_t>(int8_t *, int8_t *, int8_t *, dim3, dim3, dim3);

template __global__ void convolution3d<int16_t>(int16_t *, int16_t *, int16_t *, dim3, dim3, dim3);

template __global__ void convolution3d<int32_t>(int32_t *, int32_t *, int32_t *, dim3, dim3, dim3);

template __global__ void convolution3d<int64_t>(int64_t *, int64_t *, int64_t *, dim3, dim3, dim3);

template __global__ void convolution3d<uint8_t>(uint8_t *, uint8_t *, uint8_t *, dim3, dim3, dim3);

template __global__ void convolution3d<uint16_t>(uint16_t *, uint16_t *, uint16_t *, dim3, dim3, dim3);

template __global__ void convolution3d<uint32_t>(uint32_t *, uint32_t *, uint32_t *, dim3, dim3, dim3);

template __global__ void convolution3d<uint64_t>(uint64_t *, uint64_t *, uint64_t *, dim3, dim3, dim3);

template void convolution3d_wrapper<float>(float *, float *, float *, dim3, dim3, dim3, size_t, size_t, size_t);

template void convolution3d_wrapper<double>(double *, double *, double *, dim3, dim3, dim3, size_t, size_t, size_t);

template void convolution3d_wrapper<int8_t>(int8_t *, int8_t *, int8_t *, dim3, dim3, dim3, size_t, size_t, size_t);

template void convolution3d_wrapper<int16_t>(int16_t *, int16_t *, int16_t *, dim3, dim3, dim3, size_t, size_t, size_t);

template void convolution3d_wrapper<int32_t>(int32_t *, int32_t *, int32_t *, dim3, dim3, dim3, size_t, size_t, size_t);

template void convolution3d_wrapper<int64_t>(int64_t *, int64_t *, int64_t *, dim3, dim3, dim3, size_t, size_t, size_t);

template void convolution3d_wrapper<uint8_t>(uint8_t *, uint8_t *, uint8_t *, dim3, dim3, dim3, size_t, size_t, size_t);

template void
convolution3d_wrapper<uint16_t>(uint16_t *, uint16_t *, uint16_t *, dim3, dim3, dim3, size_t, size_t, size_t);

template void
convolution3d_wrapper<uint32_t>(uint32_t *, uint32_t *, uint32_t *, dim3, dim3, dim3, size_t, size_t, size_t);

template void
convolution3d_wrapper<uint64_t>(uint64_t *, uint64_t *, uint64_t *, dim3, dim3, dim3, size_t, size_t, size_t);


template<typename T>
__global__ void convolution3d(T *input, T *kernel, T *output, dim3 input_shape, dim3 kernel_shape, dim3 output_shape) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem_proxy[];
    T *shared_mem = reinterpret_cast<T *>(shared_mem_proxy);

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= output_shape.x || y >= output_shape.y || z >= output_shape.z)
        return;

    int out_index = z * output_shape.y * output_shape.x + y * output_shape.x + x;
    output[out_index] = 0;

    if (threadIdx.x < kernel_shape.x && threadIdx.y < kernel_shape.y && threadIdx.z < kernel_shape.z) {
        shared_mem[threadIdx.z * kernel_shape.y * kernel_shape.x + threadIdx.y * kernel_shape.x + threadIdx.x] = kernel[
                threadIdx.z * kernel_shape.y * kernel_shape.x + threadIdx.y * kernel_shape.x + threadIdx.x];
    }

    __syncthreads();

    for (int i = 0; i < kernel_shape.z; ++i) {
        for (int j = 0; j < kernel_shape.y; ++j) {
            for (int k = 0; k < kernel_shape.x; ++k) {
                int in_x = x + k;
                int in_y = y + j;
                int in_z = z + i;
                output[out_index] += input[in_z * input_shape.y * input_shape.x + in_y * input_shape.x + in_x] *
                                     shared_mem[i * kernel_shape.y * kernel_shape.x + j * kernel_shape.x + k];
            }
        }
    }
}

template<typename T>
void convolution3d_wrapper(T *input, T *kernel, T *output, dim3 input_shape, dim3 kernel_shape, dim3 output_shape,
                           size_t input_size, size_t kernel_size, size_t output_size) {
    T *d_input, *d_kernel, *d_output;
    cudaMalloc(reinterpret_cast<void **>(&d_input), input_size * sizeof(T));
    cudaMalloc(reinterpret_cast<void **>(&d_kernel), kernel_size * sizeof(T));
    cudaMalloc(reinterpret_cast<void **>(&d_output), output_size * sizeof(T));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_input, input, input_size * sizeof(T), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_kernel, kernel, kernel_size * sizeof(T), cudaMemcpyHostToDevice, stream);

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 numBlocks(
            (output_shape.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (output_shape.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (output_shape.z + threadsPerBlock.z - 1) / threadsPerBlock.z
    );
    size_t sharedMemSize = kernel_size * sizeof(T);
    convolution3d<<<numBlocks, threadsPerBlock, sharedMemSize, stream>>>(d_input, d_kernel, d_output, input_shape,
                                                                         kernel_shape, output_shape);
    cudaStreamSynchronize(stream);

    cudaMemcpyAsync(output, d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost, stream);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    cudaStreamDestroy(stream);
}