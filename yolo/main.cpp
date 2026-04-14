/**
 * @file main.cpp
 * @brief YOLOv8 demo: RPP-only CLI, preprocessing, decode, NMS, and visualization.
 *
 * Reading guide for integrators:
 *   1. **Constants** — input size 640, score/NMS thresholds (tune for your model).
 *   2. **RPP path** — `detect_rpp()` → `Yolo` + `letterbox_blob_yolov8()` → `decode_yolov8_output()` + `nms_boxes()`.
 *   3. **Output layout** — Ultralytics YOLOv8 ONNX typically exports `[1, C, N]` with C = 4 + num_classes (e.g. 84) and N anchors;
 *      values are stored in **planar** form: channel `c` at indices `c * N + i` for anchor `i` (see `decode_yolov8_output`).
 *   4. **main** — argparse, RPP inference, draw boxes and save `output.jpg`.
 *
 * Export reference: https://docs.ultralytics.com/modes/export/
 */

#include "yolo.h"
#include "argparse/argparse.hpp"
#include "utils.hpp"
#include "logger.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char* DEFAULT_IMAGE_PATH = "doc/images/test.jpg";

// Default model spatial size (must match exported ONNX input if you use 640 export).
const float INPUT_WIDTH = 640.0F;
const float INPUT_HEIGHT = 640.0F;
// Per-class score gate before NMS (Ultralytics defaults are often ~0.25).
const float SCORE_THRESHOLD = 0.25F;
// NMS IoU threshold.
const float NMS_THRESHOLD = 0.45F;

/** One final detection after NMS, in original image pixel coordinates. */
struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

const std::vector<std::string>& coco80_class_labels()
{
    static const std::vector<std::string> kLabels = {
        "person",        "bicycle",       "car",           "motorcycle",    "airplane",      "bus",           "train",         "truck",
        "boat",          "traffic light", "fire hydrant",  "stop sign",     "parking meter", "bench",         "bird",          "cat",
        "dog",           "horse",         "sheep",         "cow",           "elephant",      "bear",          "zebra",         "giraffe",
        "backpack",      "umbrella",      "handbag",       "tie",           "suitcase",      "frisbee",       "skis",          "snowboard",
        "sports ball",   "kite",          "baseball bat",  "baseball glove", "skateboard",   "surfboard",     "tennis racket", "bottle",
        "wine glass",    "cup",           "fork",          "knife",         "spoon",         "bowl",          "banana",        "apple",
        "sandwich",      "orange",        "broccoli",      "carrot",        "hot dog",       "pizza",         "donut",         "cake",
        "chair",         "couch",         "potted plant",  "bed",           "dining table",  "toilet",        "tv",            "laptop",
        "mouse",         "remote",        "keyboard",      "cell phone",    "microwave",     "oven",          "toaster",       "sink",
        "refrigerator",  "book",          "clock",         "vase",          "scissors",      "teddy bear",    "hair drier",    "toothbrush",
    };
    return kLabels;
}

std::string detection_class_label(int class_id, const std::vector<std::string>& names)
{
    if (class_id >= 0 && class_id < static_cast<int>(names.size()))
        return names[static_cast<size_t>(class_id)];
    return "class_" + std::to_string(class_id);
}

const std::vector<cv::Scalar> colors = {
    cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)
};

/** Map engine output dims to (channels C, anchors N) for 2D or batched-3D tensors. */
void yolov8_output_shape(const infer1::Dims& od, int& out_c, int& out_n)
{
    if (od.nbDims == 2) {
        out_c = od.d[0];
        out_n = od.d[1];
    } else if (od.nbDims == 3) {
        out_c = od.d[1];
        out_n = od.d[2];
    } else {
        out_c = 0;
        out_n = 0;
    }
}

/**
 * Letterbox resize (aspect preserved), pad to WxH, normalize to [0,1], RGB, layout NCHW float blob for `Yolo::infer`.
 */
cv::Mat letterbox_blob_yolov8(const cv::Mat& source, int target_w, int target_h)
{
    const float ratio = std::min(
        static_cast<float>(target_w) / static_cast<float>(source.cols),
        static_cast<float>(target_h) / static_cast<float>(source.rows));
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(), ratio, ratio);
    cv::Mat padded = cv::Mat::zeros(target_h, target_w, CV_8UC3);
    resized.copyTo(padded(cv::Rect(0, 0, resized.cols, resized.rows)));

    cv::Mat image_float;
    padded.convertTo(image_float, CV_32FC3, 1.0F / 255.0F);
    cv::cvtColor(image_float, image_float, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> channels(3);
    cv::split(image_float, channels);

    const int dims[] = { 1, 3, target_h, target_w };
    cv::Mat blob(4, dims, CV_32F);
    for (int c = 0; c < 3; ++c) {
        cv::Mat plane(target_h, target_w, CV_32F, blob.ptr(0, c));
        channels[c].copyTo(plane);
    }
    return blob;
}

/** Greedy IoU NMS: keeps high-score boxes and drops overlaps above `nms_threshold`. */
void nms_boxes(const std::vector<cv::Rect>& bboxes,
               const std::vector<float>& scores,
               float score_threshold,
               float nms_threshold,
               std::vector<int>& indices,
               float eta = 1.0f,
               int top_k = 0)
{
    indices.clear();
    if (bboxes.empty() || scores.empty() || bboxes.size() != scores.size()) {
        return;
    }

    std::vector<int> candidate_indices;
    for (int i = 0; i < static_cast<int>(scores.size()); i++) {
        if (scores[i] >= score_threshold) {
            candidate_indices.push_back(i);
        }
    }
    if (candidate_indices.empty()) {
        return;
    }

    std::sort(candidate_indices.begin(), candidate_indices.end(),
              [&scores](int a, int b) {
                  return scores[a] > scores[b];
              });

    if (top_k > 0 && top_k < static_cast<int>(candidate_indices.size())) {
        candidate_indices.resize(top_k);
    }

    std::vector<float> areas;
    for (const auto& rect : bboxes) {
        areas.push_back(static_cast<float>(rect.width * rect.height));
    }

    float adaptive_threshold = nms_threshold;

    while (!candidate_indices.empty()) {
        int best_idx = candidate_indices[0];
        indices.push_back(best_idx);

        if (candidate_indices.size() == 1) {
            break;
        }

        std::vector<int> remaining_indices;
        remaining_indices.reserve(candidate_indices.size() - 1);

        const cv::Rect& best_rect = bboxes[best_idx];

        for (size_t i = 1; i < candidate_indices.size(); i++) {
            int idx = candidate_indices[i];
            const cv::Rect& current_rect = bboxes[idx];

            int x1 = std::max(best_rect.x, current_rect.x);
            int y1 = std::max(best_rect.y, current_rect.y);
            int x2 = std::min(best_rect.x + best_rect.width, current_rect.x + current_rect.width);
            int y2 = std::min(best_rect.y + best_rect.height, current_rect.y + current_rect.height);

            float intersection = 0.0f;
            if (x2 > x1 && y2 > y1) {
                intersection = static_cast<float>((x2 - x1) * (y2 - y1));
            }

            float union_area = areas[best_idx] + areas[idx] - intersection;
            float iou = intersection / union_area;

            if (iou <= adaptive_threshold) {
                remaining_indices.push_back(idx);
            }
        }

        candidate_indices = std::move(remaining_indices);

        if (eta != 1.0f) {
            adaptive_threshold *= eta;
        }
    }
}

/**
 * Decode planar YOLOv8 head: rows are anchors, channels 0–3 box (cx,cy,w,h) in letterbox space, 4+ class scores.
 * Maps centers to top-left `cv::Rect` in original image space using the same scale as letterbox preprocessing.
 */
void decode_yolov8_output(const std::vector<float>& output,
                          int out_c,
                          int out_n,
                          const cv::Mat& original_bgr,
                          std::vector<Detection>& out)
{
    out.clear();
    if (out_c < 5 || out_n <= 0 || output.size() < static_cast<size_t>(out_c) * static_cast<size_t>(out_n)) {
        return;
    }

    float ratio_w = static_cast<float>(original_bgr.cols) / INPUT_WIDTH;
    float ratio_h = static_cast<float>(original_bgr.rows) / INPUT_HEIGHT;
    const float ratio = std::max(ratio_w, ratio_h);

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < out_n; ++i) {
        const float cx = output[i];
        const float cy = output[out_n + i];
        const float ow = output[2 * out_n + i];
        const float oh = output[3 * out_n + i];

        int best_class_id = 0;
        float best_score = 0.0F;
        for (int c = 4; c < out_c; ++c) {
            const float score = output[c * out_n + i];
            if (score > best_score) {
                best_score = score;
                best_class_id = c - 4;
            }
        }

        if (best_score <= SCORE_THRESHOLD) {
            continue;
        }

        const int x = static_cast<int>(cx * ratio);
        const int y = static_cast<int>(cy * ratio);
        const int bw = static_cast<int>(ow * ratio);
        const int bh = static_cast<int>(oh * ratio);
        const int left = x - bw / 2;
        const int top = y - bh / 2;

        class_ids.push_back(best_class_id);
        confidences.push_back(best_score);
        boxes.push_back(cv::Rect(left, top, bw, bh));
    }

    std::vector<int> nms_result;
    nms_boxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

    for (int idx : nms_result) {
        Detection det;
        det.class_id = class_ids[idx];
        det.confidence = confidences[idx];
        det.box = boxes[idx];
        out.push_back(det);
    }
}

/** Full RPP pipeline: build engine once per call */
void detect_rpp(cv::Mat& image,
                const std::string& model_path,
                std::vector<Detection>& output,
                int inference_count)
{
    Yolo yolo(model_path);
    if (!yolo.init_engine()) {
        return;
    }

    const int iw = yolo.getInputWidth();
    const int ih = yolo.getInputHeight();
    cv::Mat blob = letterbox_blob_yolov8(image, iw, ih);

    std::vector<float> output_data;
    if (!yolo.infer(blob, inference_count, output_data)) {
        return;
    }

    int out_c = 0;
    int out_n = 0;
    yolov8_output_shape(yolo.getOutputDimensions(), out_c, out_n);
    decode_yolov8_output(output_data, out_c, out_n, image, output);
}

std::string resolve_image_path(const std::string& requested_path)
{
    if (requested_path.empty()) {
        return requested_path;
    }

    if (std::filesystem::exists(requested_path)) {
        return requested_path;
    }

    const std::filesystem::path path(requested_path);
    if (path.is_absolute()) {
        return requested_path;
    }

    const auto current = std::filesystem::current_path();
    const std::vector<std::filesystem::path> candidates = {
        current / path,
        current.parent_path() / path,
    };

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }

    return requested_path;
}

} // namespace

int main(int argc, char **argv)
{

    argparse::ArgumentParser program("yolov8 demo", "0.1", argparse::default_arguments::help);

    program.add_argument("-o", "--onnx")
            .required()
            .help("path to the YOLOv8 ONNX model file (export via Ultralytics or your workflow).");
    program.add_argument("-i", "--image")
            .default_value(std::string(DEFAULT_IMAGE_PATH))
            .help("path of image file.");
    program.add_argument("-v", "--verbose")
            .help("show verbose log")
            .default_value(false)
            .implicit_value(true);
    program.add_argument("--loop")
            .default_value(1)
            .help("Loop inference count used for runtime timing")
            .scan<'i', int>();

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

    auto image_path = resolve_image_path(expand_user_path(program.get<std::string>("--image")));
    auto model_path = expand_user_path(program.get<std::string>("--onnx"));

    if (program["--verbose"] == true) {
        sample::gLogger.setReportableSeverity(infer1::ILogger::Severity::kVERBOSE);
    }
    else {
        sample::gLogger.setReportableSeverity(infer1::ILogger::Severity::kERROR);
    }

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Cannot found onnx model file, path: " << model_path << std::endl;
        return EXIT_FAILURE;
    }
    if (!image_path.empty() && !std::filesystem::exists(image_path)) {
        std::cerr << "Cannot found image file, path: " << image_path << std::endl;
        return EXIT_FAILURE;
    }

    sample::user_visible_stream_log("ONNX model path: ", model_path);
    sample::user_visible_stream_log("Image file path: ", image_path);

    int inference_count = program.get<int>("--loop");

    const std::vector<std::string>& class_list = coco80_class_labels();

    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<Detection> output;
    detect_rpp(frame, model_path, output, inference_count);

    // Overlay class names (COCO-80 embedded) and persist result next to the build cwd.
    int detections = static_cast<int>(output.size());
    for (int i = 0; i < detections; ++i)
    {
        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        const auto color = colors[classId % colors.size()];
        cv::rectangle(frame, box, color, 3);

        cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        std::string label = detection_class_label(classId, class_list);
        cv::putText(frame, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    std::ostringstream fps_label;
    fps_label << std::fixed << std::setprecision(2);
    std::string fps_label_str = fps_label.str();

    cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

    bool success = cv::imwrite("output.jpg", frame);

    if (success) {
        std::cout << "Image saved successfully as output.jpg" << std::endl;
    } else {
        std::cerr << "Failed to save image" << std::endl;
    }

    return EXIT_SUCCESS;
}
