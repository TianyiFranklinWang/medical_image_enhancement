cmake_minimum_required(VERSION 3.5...3.27)
project(medical_image_enhancement LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
add_subdirectory(pybind11)
include_directories(include)

# Set CUDA flags
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Define CUDA kernel
file(GLOB CUDA_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/conv3d.cu)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")
cuda_add_library(cuda_conv3d_lib STATIC ${CUDA_SRC_FILES})

# Define Pybind11 wrapper
file(GLOB PYBIND_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/medical_image_enhancement.cpp)
pybind11_add_module(medical_image_enhancement MODULE ${PYBIND_SRC_FILES})

# Link CUDA library
target_link_libraries(medical_image_enhancement PRIVATE cuda_conv3d_lib)

add_custom_command(
        TARGET medical_image_enhancement POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:medical_image_enhancement> ${CMAKE_CURRENT_SOURCE_DIR}
)