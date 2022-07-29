add_executable(resnet_tvm tvm_inference.cc)
target_include_directories(
    resnet_tvm
    PUBLIC ${PROJECT_SOURCE_DIR}/include
           $ENV{TVM_HOME}/3rdparty/dlpack/include
           $ENV{TVM_HOME}/3rdparty/dmlc-core/include
           $ENV{TVM_HOME}/include
           ${CMAKE_SOURCE_DIR}/third_party/argparse/include
)
target_link_directories(resnet_tvm PUBLIC $ENV{TVM_HOME}/build)
# disable -Wmacro-redefined
target_link_libraries(
    resnet_tvm
    tvm_runtime
    Threads::Threads
    ${CMAKE_DL_LIBS}
    ${OpenCV_LIBS}
    stdc++fs
    dl
)

add_executable(resnet_trt vanilla_trt.cc engine.cc inference.cc)
target_include_directories(
    resnet_trt
    PUBLIC ${PROJECT_SOURCE_DIR}/include
           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
           ${TensorRT_INCLUDE_DIR}
           ${TENSORRT_DIR}/samples/common
           ${CMAKE_SOURCE_DIR}/third_party/argparse/include
)

set_target_properties(resnet_trt PROPERTIES CUDA_ARCHITECTURES "75")
target_link_directories(resnet_trt PUBLIC ${TENSORRT_DIR}/lib)
target_link_libraries(
    resnet_trt
    TensorRT::TensorRT
    CUDA::cudart
    stdc++fs
    nvonnxparser
    ${OpenCV_LIBS}
    dl
)
