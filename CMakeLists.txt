cmake_minimum_required(VERSION 3.10)
project(YOLOv8_demo VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
    -g
    -O0
    -Wno-write-strings
    -Wno-error=deprecated-declarations
    -Wno-deprecated-declarations
    -Werror
)
add_link_options(-Wl,-rpath -Wl,$ORIGIN)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

set(RppRT_DIR /usr/local/rpp)

add_executable(YOLOv8_demo
    yolov8.cpp
    yolo_main.cpp
    common/infer/infer_sample_base.cpp
    common/calibrator/calibrator.cpp
    common/logger.cpp
    common/logging.cpp
)

target_include_directories(YOLOv8_demo PRIVATE
    common
    common/infer
    common/calibrator
    3rd_party
    3rd_party/linux/onnxruntime
    3rd_party/linux/onnxruntime/core/session
    ${OpenCV_INCLUDE_DIRS}
    ${RppRT_DIR}/include
)

target_link_directories(YOLOv8_demo PRIVATE
    ${RppRT_DIR}/lib
)

target_link_libraries(YOLOv8_demo PRIVATE
    ${OpenCV_LIBS}
    RppRT
    nvonnxparser
    caffeparser
    urpp
    pthread
)
