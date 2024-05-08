#pragma once
#ifndef MEDICAL_IMAGE_ENHANCEMENT_CONV3D_CUH
#define MEDICAL_IMAGE_ENHANCEMENT_CONV3D_CUH

#define THREADS_PER_BLOCK 8

template<typename T>
__global__ void convolution3d(T *input, T *kernel, T *output, dim3 input_shape, dim3 kernel_shape, dim3 output_shape);

template<typename T>
void convolution3d_wrapper(T *input, T *kernel, T *output, dim3 input_shape, dim3 kernel_shape, dim3 output_shape,
                           size_t input_size, size_t kernel_size, size_t output_size);

#endif //MEDICAL_IMAGE_ENHANCEMENT_CONV3D_CUH
