find_package(OpenCV REQUIRED)

if(${OpenCV_FOUND})
    message(STATUS "Found OpenCV ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else(${OpenCV_FOUND})
    message(STATUS "Could not support OpenCV")
endif(${OpenCV_FOUND})
