/**
 * @file main.cpp
 * @brief YOLOv8 I420 YUV frame demo.
 */
#include "argparse/argparse.hpp"
#include "demo_reporter.h"
#include "labels.h"
#include "logger.h"
#include "perf_trace_session.h"
#include "utils.hpp"
#include "visualization.h"
#include "yolo_pipeline.h"

#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{
#ifdef YOLO_ENABLE_RPP_PERF
constexpr bool kPerfTraceEnabled = true;
#else
constexpr bool kPerfTraceEnabled = false;
#endif
constexpr const char* kPerfTraceDir = "../trace";

/**
 * @brief Average accumulated stage timings across measured loops.
 * @param total Sum of all measured loop timings.
 * @param count Number of measured loops.
 */
YoloV8StageTimes average_times(const YoloV8StageTimes& total, int count)
{
    YoloV8StageTimes average;
    // Keep count validation local so callers can safely pass the requested loop count.
    double scale = count > 0 ? 1.0 / static_cast<double>(count) : 1.0;
    // Timing fields are accumulated as doubles and scaled into per-frame averages.
    average.input_h2d_ms = total.input_h2d_ms * scale;
    // Transfer sizes are byte counters, so integer division keeps the printed size exact per loop.
    average.input_h2d_bytes = count > 0 ? total.input_h2d_bytes / static_cast<size_t>(count) : total.input_h2d_bytes;
    average.output_d2h_ms = total.output_d2h_ms * scale;
    average.output_d2h_bytes = count > 0 ? total.output_d2h_bytes / static_cast<size_t>(count) : total.output_d2h_bytes;
    average.preprocess_ms = total.preprocess_ms * scale;
    average.inference_ms = total.inference_ms * scale;
    average.postprocess_ms = total.postprocess_ms * scale;
    average.end_to_end_ms = total.end_to_end_ms * scale;
    return average;
}
}

/**
 * @brief Run the I420 YUV frame demo from CLI parsing through rendered output.
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument values.
 */
int main(int argc, char** argv)
{
    // Step 1: declare the command-line contract for this demo.
    argparse::ArgumentParser program("YOLOv8 I420 YUV application", "2.0", argparse::default_arguments::help);

    program.add_argument("-m", "--model")
            .required()
            .help("path to the YOLOv8 ONNX model file (required).");
    program.add_argument("-i", "--input")
            .default_value(std::string("../assets/test_0.yuv"))
            .help("path of one-frame I420 YUV file.");
    program.add_argument("--yuv-width")
            .default_value(1280)
            .help("I420 YUV frame width.")
            .scan<'i', int>();
    program.add_argument("--yuv-height")
            .default_value(720)
            .help("I420 YUV frame height.")
            .scan<'i', int>();
    program.add_argument("-o", "--output")
            .default_value(std::string("../output/output_yuv.jpg"))
            .help("path of visualization output image.");
    program.add_argument("-l", "--loop")
            .default_value(1)
            .help("measured full-pipeline loop count after one warmup pass.")
            .scan<'i', int>();
    program.add_argument("-v", "--verbose")
            .help("show verbose runtime log.")
            .default_value(false)
            .implicit_value(true);

    // Step 2: normalize command-line spelling before argparse parses the options.
    auto preprocessed_arguments = preprocess_args(argc, argv);
    std::vector<const char*> fixed_arguments;
    to_char_argument_vector(preprocessed_arguments, argv, fixed_arguments);
    try {
        program.parse_args(static_cast<int>(fixed_arguments.size()), fixed_arguments.data());
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program << std::endl;
        return EXIT_FAILURE;
    }

    // Step 3: read parsed options and expand user-friendly paths such as ~/frame.yuv.
    std::string model_path = expand_user_path(program.get<std::string>("--model"));
    std::string yuv_path = expand_user_path(program.get<std::string>("--input"));
    std::string output_path = expand_user_path(program.get<std::string>("--output"));
    int yuv_width = program.get<int>("--yuv-width");
    int yuv_height = program.get<int>("--yuv-height");
    int repeat_count = program.get<int>("--loop");
    if (repeat_count <= 0) {
        std::cerr << "--loop must be greater than 0." << std::endl;
        return EXIT_FAILURE;
    }
    if (yuv_width <= 0 || yuv_height <= 0 || (yuv_width % 2) != 0 || (yuv_height % 2) != 0) {
        std::cerr << "--yuv-width and --yuv-height must be positive even values for I420 input." << std::endl;
        return EXIT_FAILURE;
    }

    // Step 4: keep runtime logs quiet by default, but expose verbose OpenRT logs on request.
    sample::gLogger.setReportableSeverity(program["--verbose"] == true ?
                                          infer1::ILogger::Severity::kVERBOSE :
                                          infer1::ILogger::Severity::kERROR);

    // Step 5: validate all files before model build so failures are immediate and readable.
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Cannot found onnx model file, path: " << model_path << std::endl;
        return EXIT_FAILURE;
    }
    if (!std::filesystem::exists(yuv_path)) {
        std::cerr << "Cannot found yuv file, path: " << yuv_path << std::endl;
        return EXIT_FAILURE;
    }

    demo_reporter::print_stage("LOAD", "Reading I420 YUV input frame");

    // I420 stores full-size Y plus quarter-size U and V planes.
    size_t yuv_bytes = static_cast<size_t>(yuv_width) * static_cast<size_t>(yuv_height) * 3U / 2U;
    std::vector<unsigned char> yuv_data(yuv_bytes);
    std::ifstream yuv_file(yuv_path, std::ios::binary);
    if (!yuv_file.read(reinterpret_cast<char*>(yuv_data.data()), static_cast<std::streamsize>(yuv_data.size()))) {
        std::cerr << "Failed to read one I420 frame from: " << yuv_path << std::endl;
        return EXIT_FAILURE;
    }

    // This host-side conversion is only for rendering the final boxes on a viewable image.
    cv::Mat yuv_mat(yuv_height * 3 / 2, yuv_width, CV_8UC1, yuv_data.data());
    cv::Mat frame;
    cv::cvtColor(yuv_mat, frame, cv::COLOR_YUV2BGR_I420);

    // Step 6: start rpp_perf automatically in perf-enabled builds so init and run events share one trace.
    PerfTraceSession trace_session("yolov8_yuv_frame_demo", kPerfTraceEnabled, kPerfTraceDir);
    // Enable driver-side trace records only when this build owns an active perf session.
    trace_session.enableDriverTrace(0);

    // Step 7: initialize the reusable YOLOv8 pipeline around the requested model.
    YoloV8PipelineOptions pipeline_options;
    pipeline_options.model_path = model_path;
    pipeline_options.inference_count = 1;

    YoloV8Pipeline pipeline;
    const std::vector<std::string>& class_list = coco80_class_labels();
    demo_reporter::print_stage("INIT", "Building the in-memory OpenRT engine on the RPP accelerator");
    if (!pipeline.init(pipeline_options, class_list)) {
        return EXIT_FAILURE;
    }

    // Run one complete pass before measurement so printed latency reflects warmed execution.
    std::vector<Detection> warmup_output;
    demo_reporter::print_stage("WARMUP", "Running 1 unmeasured frame for stable performance");
    {
        PERF_SCOPE_CATE("demo_warmup_run", "yolov8");
        if (!pipeline.runI420(yuv_data.data(), yuv_data.size(), yuv_width, yuv_height, warmup_output, nullptr)) {
            return EXIT_FAILURE;
        }
    }

    // Step 8: run the measured loop and accumulate stage timings for a stable average.
    std::vector<Detection> output;
    YoloV8StageTimes total_times;
    demo_reporter::print_stage("BENCH", "Measuring " + std::to_string(repeat_count) + " full-pipeline frames");
    auto progress_start = demo_reporter::Clock::now();
    for (int i = 0; i < repeat_count; ++i) {
        PERF_SCOPE_CATE("demo_measured_loop", "yolov8");
        YoloV8StageTimes loop_times;
        if (!pipeline.runI420(yuv_data.data(), yuv_data.size(), yuv_width, yuv_height, output, &loop_times)) {
            return EXIT_FAILURE;
        }
        total_times.input_h2d_ms += loop_times.input_h2d_ms;
        total_times.input_h2d_bytes += loop_times.input_h2d_bytes;
        total_times.output_d2h_ms += loop_times.output_d2h_ms;
        total_times.output_d2h_bytes += loop_times.output_d2h_bytes;
        total_times.preprocess_ms += loop_times.preprocess_ms;
        total_times.inference_ms += loop_times.inference_ms;
        total_times.postprocess_ms += loop_times.postprocess_ms;
        total_times.end_to_end_ms += loop_times.end_to_end_ms;
        demo_reporter::print_progress(i + 1, repeat_count, progress_start);
    }

    // Step 9: render the last measured detections onto the BGR view of the input YUV frame.
    draw_detections(frame, output, class_list);
    if (!save_detection_image(output_path, frame)) {
        std::cerr << "Failed to save image: " << output_path << std::endl;
        return EXIT_FAILURE;
    }

    demo_reporter::print_summary(average_times(total_times, repeat_count), repeat_count, output.size(), output_path);
    return EXIT_SUCCESS;
}
