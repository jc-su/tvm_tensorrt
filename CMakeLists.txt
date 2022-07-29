cmake_minimum_required(VERSION 3.21)
project(MAP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set TensorRT path
set(TENSORRT_DIR ${CMAKE_SOURCE_DIR}/third_party/TensorRT-8.0.1.6)

# set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Release with debug infomation
set(CMAKE_BUILD_TYPE ReleaseWithDebugInfo)
# optimized release
add_compile_options(-O3)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

include(FindOpenCV)

find_package(TensorRT REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)
