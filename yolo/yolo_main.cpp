/**
 * @file yolo_main.cpp
 * @brief CLI entry and CPU ONNXRuntime inference path for the YOLO demo. See README.md for full workflow.
 */
#include "yolov8.h"
#include "logger.h"
#include "argparse/argparse.hpp"
#include "utils.hpp"
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <numeric>

namespace {

struct CpuDetectRes {
    float x;
    float y;
    float w;
    float h;
    float prob;
    int classes;
};

/**
 * @brief Convert a string to lowercase for case-insensitive checks.
 * @param s Input string.
 * @return Lowercase string.
 */
std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

/**
 * @brief Detect whether model filename indicates INT8 quantization.
 * @param model_path ONNX model path.
 * @return true if filename contains "int8" or "quant".
 */
bool is_int8_model(const std::string& model_path) {
    const std::string lower = to_lower(std::filesystem::path(model_path).stem().string());
    return lower.find("int8") != std::string::npos || lower.find("quant") != std::string::npos;
}

/**
 * @brief Compute IoU for two center-format bounding boxes.
 * @param det_a First detection box.
 * @param det_b Second detection box.
 * @return IoU value in [0, 1].
 */
float iou_calculate(const CpuDetectRes& det_a, const CpuDetectRes& det_b) {
    const float inter_l = std::max(det_a.x - det_a.w / 2.0F, det_b.x - det_b.w / 2.0F);
    const float inter_t = std::max(det_a.y - det_a.h / 2.0F, det_b.y - det_b.h / 2.0F);
    const float inter_r = std::min(det_a.x + det_a.w / 2.0F, det_b.x + det_b.w / 2.0F);
    const float inter_b = std::min(det_a.y + det_a.h / 2.0F, det_b.y + det_b.h / 2.0F);

    if (inter_b < inter_t || inter_r < inter_l) {
        return 0.0F;
    }

    const float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    const float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area <= 0.0F) {
        return 0.0F;
    }
    return inter_area / union_area;
}

/**
 * @brief Decode output tensor and apply score filtering plus class-wise NMS.
 * @param output Flat tensor values from model output.
 * @param out_h Output channel dimension.
 * @param out_w Output candidate dimension.
 * @param img Original image used for coordinate scaling.
 * @return Final detection list.
 */
std::vector<CpuDetectRes> nms_from_output(const std::vector<float>& output, int out_h, int out_w, const cv::Mat& img) {
    constexpr float kScoreThreshold = 0.25F;
    constexpr float kIouThreshold = 0.45F;
    constexpr int kInputWidth = 640;
    constexpr int kInputHeight = 640;

    std::vector<CpuDetectRes> result;
    float ratio_w = static_cast<float>(img.cols) / static_cast<float>(kInputWidth);
    float ratio_h = static_cast<float>(img.rows) / static_cast<float>(kInputHeight);
    const float ratio = std::max(ratio_w, ratio_h);
    ratio_w = ratio_h = ratio;

    // Output layout is [1, C, N]; evaluate each candidate and keep best class score.
    for (int i = 0; i < out_w; ++i) {
        const float cx = output[i];
        const float cy = output[out_w + i];
        const float ow = output[2 * out_w + i];
        const float oh = output[3 * out_w + i];

        int best_class_id = 0;
        float best_score = 0.0F;
        for (int c = 4; c < out_h; ++c) {
            const float score = output[c * out_w + i];
            if (score > best_score) {
                best_score = score;
                best_class_id = c - 4;
            }
        }

        if (best_score <= kScoreThreshold) {
            continue;
        }

        CpuDetectRes box {};
        box.x = cx * ratio_w;
        box.y = cy * ratio_h;
        box.w = ow * ratio_w;
        box.h = oh * ratio_h;
        box.prob = best_score;
        box.classes = best_class_id;
        result.emplace_back(box);
    }

    std::sort(result.begin(), result.end(), [](const CpuDetectRes& left, const CpuDetectRes& right) {
        return left.prob > right.prob;
    });

    for (size_t i = 0; i < result.size(); ++i) {
        if (result[i].prob == 0.0F) {
            continue;
        }
        for (size_t j = i + 1; j < result.size(); ++j) {
            if (result[j].prob == 0.0F || result[i].classes != result[j].classes) {
                continue;
            }
            if (iou_calculate(result[i], result[j]) > kIouThreshold) {
                result[j].prob = 0.0F;
            }
        }
    }

    result.erase(std::remove_if(result.begin(), result.end(), [](const CpuDetectRes& det) {
        return det.prob == 0.0F;
    }), result.end());

    return result;
}

/**
 * @brief Run YOLO inference on CPU with ONNXRuntime and draw confidence labels.
 * @param model_path ONNX model path.
 * @param image_path Input image path.
 * @param loops Number of timed inference runs.
 * @return true on success, false on failure.
 */
bool infer_with_cpu_onnxruntime(const std::string& model_path, const std::string& image_path, int loops) {
    // Load input image from local path.
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        sample::LOG_ERROR() << "Could not open or find the image: " << image_path << std::endl;
        return false;
    }

    // Resize with aspect-ratio preservation and pad to model input size.
    constexpr int kInputWidth = 640;
    constexpr int kInputHeight = 640;
    float ratio = std::min(
        static_cast<float>(kInputWidth) / static_cast<float>(image.cols),
        static_cast<float>(kInputHeight) / static_cast<float>(image.rows)
    );
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(), ratio, ratio);
    cv::Mat padded = cv::Mat::zeros(cv::Size(kInputWidth, kInputHeight), CV_8UC3);
    resized.copyTo(padded(cv::Rect(0, 0, resized.cols, resized.rows)));

    // Convert image to normalized RGB NCHW tensor.
    cv::Mat image_float;
    padded.convertTo(image_float, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(image_float, image_float, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> channels(3);
    cv::split(image_float, channels);

    std::vector<float> input_tensor_values(3 * kInputWidth * kInputHeight);
    const size_t channel_size = static_cast<size_t>(kInputWidth * kInputHeight);
    std::memcpy(input_tensor_values.data(), channels[0].data, channel_size * sizeof(float));
    std::memcpy(input_tensor_values.data() + channel_size, channels[1].data, channel_size * sizeof(float));
    std::memcpy(input_tensor_values.data() + channel_size * 2, channels[2].data, channel_size * sizeof(float));

    // ONNXRuntime: create CPU session and runtime allocator.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolov8_cpu_demo");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto onnx_output_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<int64_t> input_shape {1, 3, kInputHeight, kInputWidth};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
    );
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {onnx_output_name.get()};

    // Warm up once so first timed iteration is more stable.
    (void)session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // Measure post-warmup inference latency.
    std::vector<double> cost_list_ms;
    std::vector<Ort::Value> output_tensors;
    for (int i = 0; i < loops; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        const auto end = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        cost_list_ms.emplace_back(ms);
    }

    if (output_tensors.empty()) {
        sample::LOG_ERROR() << "ONNXRuntime returns empty outputs." << std::endl;
        return false;
    }

    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.size() != 3) {
        sample::LOG_ERROR() << "Unexpected output shape rank: " << output_shape.size() << std::endl;
        return false;
    }
    const int out_h = static_cast<int>(output_shape[1]);
    const int out_w = static_cast<int>(output_shape[2]);
    const size_t output_size = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
    const float* output_data_ptr = output_tensors[0].GetTensorData<float>();
    std::vector<float> output(output_data_ptr, output_data_ptr + output_size);

    // Decode detections and draw confidence on output image.
    cv::Mat draw_image = image.clone();
    const auto detections = nms_from_output(output, out_h, out_w, draw_image);
    for (const auto& box : detections) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.1f%%", box.prob * 100.0);
        const cv::Rect rect(
            static_cast<int>(box.x - box.w / 2.0F),
            static_cast<int>(box.y - box.h / 2.0F),
            static_cast<int>(box.w),
            static_cast<int>(box.h)
        );
        cv::rectangle(draw_image, rect, cv::Scalar(0, 255, 0), 2, cv::LINE_8, 0);

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        cv::Point text_org(rect.x, rect.y - 4);
        if (text_org.y - text_size.height < 0) text_org.y = rect.y + text_size.height + 4;
        cv::rectangle(draw_image, cv::Point(text_org.x, text_org.y - text_size.height - 4),
                      cv::Point(text_org.x + text_size.width, text_org.y + 4),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(draw_image, buf, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
    const std::string output_path = "./output_cpu.jpg";
    cv::imwrite(output_path, draw_image);

    if (!cost_list_ms.empty()) {
        const double total_cost = std::accumulate(cost_list_ms.begin(), cost_list_ms.end(), 0.0);
        const double avg_cost = total_cost / static_cast<double>(cost_list_ms.size());
        sample::user_visible_stream_log(
            "CPU inference [", cost_list_ms.size(), "] times, total takes: ", total_cost,
            " milliseconds, average inference time: ", avg_cost,
            ", frames per second: ", static_cast<int>(1000.0 / avg_cost)
        );
    }
    sample::user_visible_stream_log("output image: ", output_path);
    return true;
}

} // namespace

/**
 * @brief Parse CLI options, select CPU/RPP path, and run one-image YOLO inference.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success, non-zero on failure.
 */
int main(int argc, char *argv[])
{
    // Parse runtime options for model/image/device/loop settings.
    argparse::ArgumentParser program("Yolov8 demo", "0.1", argparse::default_arguments::help);

    program.add_argument("-o", "--onnx")
            .default_value(std::string("yolov8m.onnx"))
            .help("path of onnx model file (auto-detects variant and precision).");
    program.add_argument("-i", "--image")
            .default_value(std::string(""))
            .help("path of image file.");
    program.add_argument("-v", "--verbose")
            .help("show verbose log")
            .default_value(false)
            .implicit_value(true);
    program.add_argument("-l", "--loop")
            .default_value(1)
            .help("Loop inference count")
            .scan<'i', int>();
    program.add_argument("-d", "--device")
            .default_value(std::string("rpp"))
            .help("inference device: rpp or cpu");

    // Keep compatibility for "-x=value" style arguments.
    auto preprocessed_arguments = preprocess_args(argc, argv);
    std::vector<const char*> fixed_arguments;
    to_char_argument_vector(preprocessed_arguments, argv, fixed_arguments);
    try {
        program.parse_args(int(fixed_arguments.size()), fixed_arguments.data());
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Configure logger verbosity based on CLI switch.
    if (program["--verbose"] == true) {
        sample::gLogger.setReportableSeverity(infer1::ILogger::Severity::kVERBOSE);
    } else {
        sample::gLogger.setReportableSeverity(infer1::ILogger::Severity::kERROR);
    }

    // Resolve normalized runtime parameters.
    const auto model_path = expand_user_path(program.get<std::string>("--onnx"));
    auto image_source = expand_user_path(program.get<std::string>("--image"));
    const std::string device = to_lower(program.get<std::string>("--device"));
    const int loops = std::max(1, program.get<int>("--loop"));

    if (device != "rpp" && device != "cpu") {
        std::cerr << "Invalid --device value: " << device << ". Expected 'rpp' or 'cpu'." << std::endl;
        return EXIT_FAILURE;
    }

    // Validate required input files before inference.
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Cannot find onnx model file: " << model_path << std::endl;
        return EXIT_FAILURE;
    }

    // Infer precision mode from filename convention.
    const bool use_int8 = is_int8_model(model_path);
    sample::user_visible_stream_log("Detected precision: ", use_int8 ? "int8" : "bf16");

    // Use a default demo image when user does not pass --image.
    if (image_source.empty()) {
        image_source = "../doc/images/5461697264_b231724778_b.jpg";
    }
    if (!std::filesystem::exists(image_source)) {
        std::cerr << "Cannot find image file: " << image_source << std::endl;
        return EXIT_FAILURE;
    }
    // CPU path runs pure ONNXRuntime inference.
    if (device == "cpu") {
        sample::LOG_INFO() << "Running with CPU (ONNX Runtime)." << std::endl;
        const bool ok = infer_with_cpu_onnxruntime(model_path, image_source, loops);
        return ok ? 0 : -1;
    }

    // RPP path builds engine and runs full pipeline.
    samplesCommon::ModelSampleParams params;
    params.loops = loops;
    params.onnxFileName = model_path;
    params.inferImageFileName = image_source;
    params.image_count = 1;
    params.int8 = use_int8;
    params.bf16 = !use_int8;

    Yolov8s sample(params);
    sample::LOG_INFO() << "Building and running RPP inference." << std::endl;

    if (!sample.build()) {
        sample::LOG_ERROR() << "Build failed" << std::endl;
        return -1;
    }
    if (!sample.infer()) {
        sample::LOG_ERROR() << "Infer failed" << std::endl;
        return -1;
    }

    sample.teardown();
    return 0;
}
