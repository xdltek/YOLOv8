# YOLOv8 demo: OpenCV + RppRT.
cmake_minimum_required(VERSION 3.10)
project(yolov8_demo_workspace VERSION 1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wno-write-strings -Wno-error=deprecated-declarations -Wno-deprecated-declarations -Werror")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

set(RPP_DIR /usr/local/rpp)
link_directories(${RPP_DIR}/lib)

add_executable(yolov8_demo
    yolo/main.cpp
    yolo/yolo.cpp
    common/logger.cpp
    common/logging.cpp
)

target_include_directories(yolov8_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/yolo
    ${RPP_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolov8_demo PRIVATE
    ${OpenCV_LIBS}
    RppRT nvonnxparser caffeparser urpp
    pthread
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/image/test_1.png")
    configure_file(image/test_1.png "${CMAKE_BINARY_DIR}/test_1.png" COPYONLY)
endif()
