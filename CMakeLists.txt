cmake_minimum_required(VERSION 3.16.3)

project(yolov8_demo_workspace VERSION 1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wno-write-strings -Wno-error=deprecated-declarations -Wno-deprecated-declarations -Werror")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

set(CMAKE_CUDA_COMPILER_FORCED TRUE)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_COMPILER /usr/local/bin/clang++)
set(CMAKE_CUDA_ARCHITECTURES RPP)
set(_CMAKE_CUDA_WHOLE_FLAG "-c")
enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "-fPIC -gdwarf-4 -fno-unroll-loops -O3 -ffp-contract=off -nocudalib -nocudainc -save-temps -std=c++14")

option(YOLO_ENABLE_RPP_PERF "Enable rpp_perf trace integration" OFF)

set(ROOT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(RppRT_DIR /usr/local/rpp)
set(RppRT_LIBS ${RppRT_DIR}/lib)
set(RppRT_INCLUDE_DIRS "${RppRT_DIR}/include")
set(ThirdPartyLib_Include ${ROOT_DIRECTORY}/3rd_party)
set(YOLO_BUILD_BIN_DIR ${CMAKE_BINARY_DIR}/bin)
set(YOLO_BUILD_LIB_DIR ${CMAKE_BINARY_DIR}/lib)
set(YOLO_BUILD_ASSETS_DIR ${CMAKE_BINARY_DIR}/assets)
set(YOLO_BUILD_OUTPUT_DIR ${CMAKE_BINARY_DIR}/output)
set(YOLO_BUILD_TRACE_DIR ${CMAKE_BINARY_DIR}/trace)
set(YOLO_BUILD_LOGS_DIR ${CMAKE_BINARY_DIR}/logs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${YOLO_BUILD_BIN_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${YOLO_BUILD_LIB_DIR})

message(STATUS "ROOT_DIRECTORY: ${ROOT_DIRECTORY}")

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
)

link_directories(${RppRT_LIBS})
set(rpprt_libs RppRT nvonnxparser caffeparser urpp)

set(YOLOV8_INCLUDE_DIR ${ROOT_DIRECTORY}/src/yolov8)
set(YOLO_UTILS_INCLUDE_DIR ${ROOT_DIRECTORY}/src/utils)
set(YOLO_PERF_INCLUDE_DIR ${ROOT_DIRECTORY}/src/perf)
set(RPP_INFERENCE_INCLUDE_DIR ${ROOT_DIRECTORY}/rpp/inference)
set(RPP_PREPROCESS_INCLUDE_DIR ${ROOT_DIRECTORY}/rpp/preprocess)
set(RPP_POSTPROCESS_INCLUDE_DIR ${ROOT_DIRECTORY}/rpp/postprocess)

add_library(yolov8_utils SHARED
    ${ROOT_DIRECTORY}/src/utils/logger.cpp
    ${ROOT_DIRECTORY}/src/utils/logging.cpp
)
target_include_directories(yolov8_utils PUBLIC
    ${YOLO_UTILS_INCLUDE_DIR}
    ${RppRT_INCLUDE_DIRS}
)
set_target_properties(yolov8_utils PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_library(perf_trace SHARED
    ${ROOT_DIRECTORY}/src/perf/perf_trace_session.cpp
)
target_include_directories(perf_trace PUBLIC
    ${YOLO_PERF_INCLUDE_DIR}
)
target_include_directories(perf_trace PRIVATE
    ${RppRT_INCLUDE_DIRS}
)
target_link_libraries(perf_trace PRIVATE pthread)
if(YOLO_ENABLE_RPP_PERF)
    target_compile_definitions(perf_trace PUBLIC YOLO_ENABLE_RPP_PERF=1)
    target_compile_options(perf_trace PRIVATE -Wno-c99-designator)
    target_link_libraries(perf_trace PRIVATE rpp_perf urpp)
endif()
set_target_properties(perf_trace PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_library(rpp_yolo_preprocess SHARED
    ${ROOT_DIRECTORY}/rpp/preprocess/rpp_yolo_preprocessor.cpp
    ${ROOT_DIRECTORY}/rpp/preprocess/rpp_preprocess_kernels.cu
)
target_include_directories(rpp_yolo_preprocess PUBLIC
    ${RPP_PREPROCESS_INCLUDE_DIR}
    ${YOLOV8_INCLUDE_DIR}
    ${RppRT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(rpp_yolo_preprocess PRIVATE
    ${YOLO_PERF_INCLUDE_DIR}
    ${YOLO_UTILS_INCLUDE_DIR}
)
target_link_libraries(rpp_yolo_preprocess PUBLIC perf_trace yolov8_utils ${rpprt_libs})
set_target_properties(rpp_yolo_preprocess PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_library(rpp_inference SHARED
    ${ROOT_DIRECTORY}/rpp/inference/rpp_infer_engine.cpp
)
target_include_directories(rpp_inference PUBLIC
    ${RPP_INFERENCE_INCLUDE_DIR}
    ${YOLO_UTILS_INCLUDE_DIR}
    ${RppRT_INCLUDE_DIRS}
)
target_link_libraries(rpp_inference PUBLIC yolov8_utils ${rpprt_libs})
set_target_properties(rpp_inference PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_library(rpp_yolov8_postprocess SHARED
    ${ROOT_DIRECTORY}/rpp/postprocess/rpp_yolov8_postprocessor.cpp
)
target_include_directories(rpp_yolov8_postprocess PUBLIC
    ${RPP_POSTPROCESS_INCLUDE_DIR}
    ${YOLOV8_INCLUDE_DIR}
    ${RppRT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(rpp_yolov8_postprocess PRIVATE
    ${YOLO_PERF_INCLUDE_DIR}
    ${YOLO_UTILS_INCLUDE_DIR}
)
target_link_libraries(rpp_yolov8_postprocess PUBLIC perf_trace yolov8_utils ${rpprt_libs} ${OpenCV_LIBS})
set_target_properties(rpp_yolov8_postprocess PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_library(yolov8_core SHARED
    ${ROOT_DIRECTORY}/src/yolov8/labels.cpp
    ${ROOT_DIRECTORY}/src/yolov8/yolo_pipeline.cpp
)
target_include_directories(yolov8_core PUBLIC
    ${YOLOV8_INCLUDE_DIR}
    ${RPP_INFERENCE_INCLUDE_DIR}
    ${RPP_POSTPROCESS_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolov8_core PUBLIC
    perf_trace
    rpp_inference
    rpp_yolo_preprocess
    rpp_yolov8_postprocess
    yolov8_utils
    ${OpenCV_LIBS}
    ${rpprt_libs}
    pthread
)
set_target_properties(yolov8_core PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_library(yolov8_visualization SHARED
    ${ROOT_DIRECTORY}/src/yolov8/visualization.cpp
)
target_include_directories(yolov8_visualization PUBLIC
    ${YOLOV8_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolov8_visualization PUBLIC yolov8_core ${OpenCV_LIBS})
set_target_properties(yolov8_visualization PROPERTIES
    BUILD_RPATH "$ORIGIN;${RppRT_LIBS}"
)

add_executable(yolov8_rgb_image_demo
    ${ROOT_DIRECTORY}/demos/rgb_image/main.cpp
)
target_include_directories(yolov8_rgb_image_demo PRIVATE
    ${ThirdPartyLib_Include}
)
target_link_libraries(yolov8_rgb_image_demo PRIVATE yolov8_core yolov8_visualization)
set_target_properties(yolov8_rgb_image_demo PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib;${RppRT_LIBS}"
)

add_executable(yolov8_yuv_frame_demo
    ${ROOT_DIRECTORY}/demos/yuv_frame/main.cpp
)
target_include_directories(yolov8_yuv_frame_demo PRIVATE
    ${ThirdPartyLib_Include}
)
target_link_libraries(yolov8_yuv_frame_demo PRIVATE yolov8_core yolov8_visualization)
set_target_properties(yolov8_yuv_frame_demo PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib;${RppRT_LIBS}"
)

file(MAKE_DIRECTORY ${YOLO_BUILD_ASSETS_DIR})
file(MAKE_DIRECTORY ${YOLO_BUILD_OUTPUT_DIR})
file(MAKE_DIRECTORY ${YOLO_BUILD_TRACE_DIR})
file(MAKE_DIRECTORY ${YOLO_BUILD_LOGS_DIR})
configure_file("${ROOT_DIRECTORY}/assets/test.jpg" "${YOLO_BUILD_ASSETS_DIR}/test.jpg" COPYONLY)
configure_file("${ROOT_DIRECTORY}/assets/test_0.yuv" "${YOLO_BUILD_ASSETS_DIR}/test_0.yuv" COPYONLY)
