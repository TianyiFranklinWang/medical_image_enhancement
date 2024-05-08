#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <iostream>
#include "conv3d.cuh"

namespace py = pybind11;

template<typename T>
py::array_t<T> convolve3d(py::array_t<T, py::array::c_style | py::array::forcecast> input, py::array_t<T, py::array::c_style | py::array::forcecast> kernel) {
    py::buffer_info input_info = input.request();
    py::buffer_info kernel_info = kernel.request();

    if (input_info.ndim != 3 || kernel_info.ndim != 3) {
        throw std::runtime_error("Number of dimensions must be 3");
    }

    std::vector<ssize_t> v_output_shape(3);
    for (int i = 0; i < 3; ++i) {
        v_output_shape[i] = input_info.shape[i] - kernel_info.shape[i] + 1;
        if (v_output_shape[i] <= 0) {
            throw std::runtime_error("Invalid input and kernel sizes");
        }
    }
    auto output = py::array_t<T>(v_output_shape);
    py::buffer_info output_info = output.request();

    T *input_data = reinterpret_cast<T *>(input_info.ptr);
    T *kernel_data = reinterpret_cast<T *>(kernel_info.ptr);
    T *output_data = reinterpret_cast<T *>(output_info.ptr);

    dim3 input_shape(input_info.shape[2], input_info.shape[1], input_info.shape[0]);
    dim3 kernel_shape(kernel_info.shape[2], kernel_info.shape[1], kernel_info.shape[0]);
    dim3 output_shape(output_info.shape[2], output_info.shape[1], output_info.shape[0]);
    size_t input_size = input_info.size;
    size_t kernel_size = kernel_info.size;
    size_t output_size = output_info.size;
    convolution3d_wrapper(input_data, kernel_data, output_data, input_shape, kernel_shape, output_shape, input_size,
                          kernel_size, output_size);

    return output;
}

// Dispatcher function to choose appropriate convolve3d based on input dtype
py::object conv3d(py::object input, py::object kernel) {
    // Determine data type of input and dispatch appropriate function
    if (py::dtype::of<float>().is(input.attr("dtype")) && py::dtype::of<float>().is(kernel.attr("dtype"))) {
        return convolve3d<float>(input, kernel);
    } else if (py::dtype::of<double>().is(input.attr("dtype")) && py::dtype::of<double>().is(kernel.attr("dtype"))) {
        return convolve3d<double>(input, kernel);
    } else if (py::dtype::of<int8_t>().is(input.attr("dtype")) && py::dtype::of<int8_t>().is(kernel.attr("dtype"))) {
        return convolve3d<int8_t>(input, kernel);
    } else if (py::dtype::of<int16_t>().is(input.attr("dtype")) && py::dtype::of<int16_t>().is(kernel.attr("dtype"))) {
        return convolve3d<int16_t>(input, kernel);
    } else if (py::dtype::of<int32_t>().is(input.attr("dtype")) && py::dtype::of<int32_t>().is(kernel.attr("dtype"))) {
        return convolve3d<int32_t>(input, kernel);
    } else if (py::dtype::of<int64_t>().is(input.attr("dtype")) && py::dtype::of<int64_t>().is(kernel.attr("dtype"))) {
        return convolve3d<int64_t>(input, kernel);
    } else if (py::dtype::of<uint8_t>().is(input.attr("dtype")) && py::dtype::of<uint8_t>().is(kernel.attr("dtype"))) {
        return convolve3d<uint8_t>(input, kernel);
    } else if (py::dtype::of<uint16_t>().is(input.attr("dtype")) &&
               py::dtype::of<uint16_t>().is(kernel.attr("dtype"))) {
        return convolve3d<uint16_t>(input, kernel);
    } else if (py::dtype::of<uint32_t>().is(input.attr("dtype")) &&
               py::dtype::of<uint32_t>().is(kernel.attr("dtype"))) {
        return convolve3d<uint32_t>(input, kernel);
    } else if (py::dtype::of<uint64_t>().is(input.attr("dtype")) &&
               py::dtype::of<uint64_t>().is(kernel.attr("dtype"))) {
        return convolve3d<uint64_t>(input, kernel);
    } else {
        throw std::runtime_error(
                "Unsupported data type. Supported types are float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, and uint64.");
    }
}

PYBIND11_MODULE(medical_image_enhancement, m) {
    m.def("conv3d", &conv3d, "Convolve 3D using CUDA");
}
