/**
 * @file rpp_yolov8_postprocessor.cpp
 * @brief YOLOv8 output copy, decode, and NMS module.
 */
#include "rpp_yolov8_postprocessor.h"

#include <InferRuntime.h>
#include <nmspre_plugin.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>

REGISTER_OPENRT_PLUGIN(NMSPREPluginCreator);

namespace
{
using ProfileClock = std::chrono::high_resolution_clock;
constexpr int kMaxOutputBoxesPerClass = 200;

/**
 * @brief Minimal logger used while building small RT_NMS helper engines.
 */
class RtnmsLogger : public infer1::ILogger
{
public:
    void log(Severity severity, const char* msg) ORTNOEXCEPT override
    {
        if (severity <= Severity::kERROR) {
            std::cerr << "[YOLOv8 RT_NMS] " << msg << std::endl;
        }
    }
};

/**
 * @brief Return the process-wide logger required by OpenRT builder APIs.
 */
RtnmsLogger& rtnms_logger()
{
    static RtnmsLogger logger;
    return logger;
}

/**
 * @brief Measure elapsed milliseconds from a saved timestamp to now.
 */
double elapsed_ms(ProfileClock::time_point start)
{
    return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}

/**
 * @brief Clear an optional postprocess profile before filling current-run timings.
 */
void reset_profile(YoloV8PostprocessProfile* profile)
{
    if (profile != nullptr) {
        *profile = YoloV8PostprocessProfile{};
    }
}

/**
 * @brief Print a runtime API error and convert the status into a boolean.
 */
bool check_rt(rtError_t status, const char* what)
{
    if (status == rtError_t::rtSuccess) {
        return true;
    }
    std::cerr << what << " failed, rtError=" << static_cast<int>(status) << std::endl;
    return false;
}

/**
 * @brief Return the element count for a static OpenRT dimension object.
 */
int dims_volume(const infer1::Dims& dims)
{
    int volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            return 0;
        }
        volume *= dims.d[i];
    }
    return volume;
}

/**
 * @brief Allocate a device DDR buffer and report failures with operation context.
 */
bool allocate_device(void** ptr, size_t bytes, const char* name)
{
    if (bytes == 0) {
        return false;
    }
    return check_rt(rtMalloc(ptr, bytes), name);
}

/**
 * @brief Allocate a pinned host buffer and report failures with operation context.
 */
bool allocate_host_pinned(void** ptr, size_t bytes, const char* name)
{
    if (bytes == 0) {
        return false;
    }
    return check_rt(rtMallocHost(ptr, bytes), name);
}

/**
 * @brief Clip floating-point corner coordinates to an OpenCV rectangle inside the source image.
 */
cv::Rect clip_rect_to_image(float left, float top, float right, float bottom, int width, int height)
{
    left = std::max(0.0f, std::min(left, static_cast<float>(width - 1)));
    top = std::max(0.0f, std::min(top, static_cast<float>(height - 1)));
    right = std::max(0.0f, std::min(right, static_cast<float>(width - 1)));
    bottom = std::max(0.0f, std::min(bottom, static_cast<float>(height - 1)));

    int x0 = static_cast<int>(left);
    int y0 = static_cast<int>(top);
    int x1 = static_cast<int>(right);
    int y1 = static_cast<int>(bottom);
    return cv::Rect(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));
}

/**
 * @brief Compute IoU for two clipped host rectangles.
 */
float intersection_over_union(const cv::Rect& a, const cv::Rect& b)
{
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    float intersection = 0.0f;
    if (x2 > x1 && y2 > y1) {
        intersection = static_cast<float>((x2 - x1) * (y2 - y1));
    }

    float union_area = static_cast<float>(a.area() + b.area()) - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

/**
 * @brief CPU fallback greedy NMS used when RT_NMS is unavailable.
 */
void nms_boxes(const std::vector<cv::Rect>& boxes,
               const std::vector<float>& scores,
               const std::vector<int>& class_ids,
               bool agnostic,
               float score_threshold,
               float nms_threshold,
               std::vector<int>& keep)
{
    keep.clear();

    std::vector<int> candidates;
    for (int i = 0; i < static_cast<int>(scores.size()); ++i) {
        if (scores[static_cast<size_t>(i)] >= score_threshold) {
            candidates.push_back(i);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [&scores](int a, int b) {
        return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)];
    });

    while (!candidates.empty()) {
        int best = candidates.front();
        keep.push_back(best);

        std::vector<int> next;
        next.reserve(candidates.size());
        for (size_t i = 1; i < candidates.size(); ++i) {
            int idx = candidates[i];
            if (!agnostic && class_ids[static_cast<size_t>(idx)] != class_ids[static_cast<size_t>(best)]) {
                next.push_back(idx);
                continue;
            }
            if (intersection_over_union(boxes[static_cast<size_t>(best)], boxes[static_cast<size_t>(idx)]) <= nms_threshold) {
                next.push_back(idx);
            }
        }
        candidates.swap(next);
    }
}
}

/**
 * @brief Release RT_NMS buffers and helper engines owned by this instance.
 */
YoloV8Postprocessor::~YoloV8Postprocessor()
{
    releaseRtnmsBuffers();
    resetRtnmsEngines();
}

/**
 * @brief Store config, resolve output layout, and try to initialize the RT_NMS path.
 */
bool YoloV8Postprocessor::init(const YoloV8Config& config)
{
    if (config.output_elements == 0 || config.score_threshold < 0.0f ||
        config.nms_threshold < 0.0f || config.nms_threshold > 1.0f) {
        std::cerr << "Invalid YOLOv8 postprocess configuration." << std::endl;
        return false;
    }

    config_ = config;
    if (!resolveOutputLayout()) {
        std::cerr << "Unsupported YOLOv8 output dimensions." << std::endl;
        return false;
    }

    host_output_.assign(config_.output_elements, 0.0f);
    // RT_NMS is an optimization; unsupported layouts fall back to the CPU path.
    rtnms_initialized_ = initRtnms();
    if (!rtnms_initialized_) {
        std::cerr << "YOLOv8 RT_NMS postprocess init failed; using CPU postprocess fallback." << std::endl;
    }

    initialized_ = true;
    return true;
}

/**
 * @brief Resolve common YOLOv8 output layouts such as [C,N], [N,C], [1,C,N], and [1,N,C].
 */
bool YoloV8Postprocessor::resolveOutputLayout()
{
    layout_ = OutputLayout{};
    for (int i = 0; i < config_.output_dimensions.nbDims; ++i) {
        if (config_.output_dimensions.d[i] > 0) {
            layout_.dims.push_back(config_.output_dimensions.d[i]);
        }
    }
    if (layout_.dims.size() < 2) {
        return false;
    }

    // Prefer the configured class count when it clearly identifies the feature axis.
    int expected_channels = config_.class_count > 0 ? config_.class_count + 4 : 0;
    int feature_axis = -1;
    for (int i = 0; i < static_cast<int>(layout_.dims.size()); ++i) {
        if (expected_channels > 4 && layout_.dims[static_cast<size_t>(i)] == expected_channels) {
            feature_axis = i;
            break;
        }
    }

    // If class count was not enough, pick the smallest non-box axis as the feature axis.
    if (feature_axis < 0) {
        int best_dim = std::numeric_limits<int>::max();
        for (int i = 0; i < static_cast<int>(layout_.dims.size()); ++i) {
            int dim = layout_.dims[static_cast<size_t>(i)];
            if (dim > 4 && dim < best_dim) {
                best_dim = dim;
                feature_axis = i;
            }
        }
    }
    if (feature_axis < 0) {
        return false;
    }

    layout_.feature_axis = feature_axis;
    layout_.channels = layout_.dims[static_cast<size_t>(feature_axis)];
    if (layout_.channels <= 4) {
        return false;
    }
    config_.class_count = layout_.channels - 4;

    layout_.strides.assign(layout_.dims.size(), 1U);
    size_t volume = 1U;
    for (int i = static_cast<int>(layout_.dims.size()) - 1; i >= 0; --i) {
        layout_.strides[static_cast<size_t>(i)] = volume;
        volume *= static_cast<size_t>(layout_.dims[static_cast<size_t>(i)]);
    }
    if (volume != config_.output_elements) {
        return false;
    }

    size_t anchors = volume / static_cast<size_t>(layout_.channels);
    if (anchors == 0 || anchors > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    layout_.anchors = static_cast<int>(anchors);
    return true;
}

bool YoloV8Postprocessor::run(const void* yolo_output_device,
                              const LetterboxInfo& letterbox,
                              std::vector<Detection>& detections,
                              rtStream_t stream,
                              YoloV8PostprocessProfile* profile)
{
    reset_profile(profile);
    detections.clear();

    if (!initialized_ || yolo_output_device == nullptr) {
        std::cerr << "YOLOv8 postprocessor has not been initialized." << std::endl;
        return false;
    }
    if (letterbox.scale <= 0.0f || letterbox.source_width <= 0 || letterbox.source_height <= 0) {
        std::cerr << "Invalid letterbox metadata for YOLOv8 postprocess." << std::endl;
        return false;
    }

    rtStream_t run_stream = stream;
    bool owns_stream = false;
    if (run_stream == nullptr && rtnms_initialized_) {
        if (!check_rt(rtStreamCreate(&run_stream), "rtStreamCreate YOLOv8 postprocess")) {
            return false;
        }
        owns_stream = true;
    }

    bool ok = false;
    if (rtnms_initialized_) {
        ok = runRtnms(yolo_output_device, letterbox, detections, run_stream, profile);
        if (!ok) {
            detections.clear();
            std::cerr << "YOLOv8 RT_NMS postprocess failed; retrying CPU postprocess fallback." << std::endl;
        }
    }
    if (owns_stream) {
        check_rt(rtStreamDestroy(run_stream), "rtStreamDestroy YOLOv8 postprocess");
    }
    if (ok) {
        return true;
    }

    return runCpuFallback(yolo_output_device, letterbox, detections, profile);
}

/**
 * @brief Validate RT_NMS-compatible output shape and build all fast-path resources.
 */
bool YoloV8Postprocessor::initRtnms()
{
    if (layout_.dims.size() < 2 || layout_.dims.size() > 3) {
        return false;
    }

    // Drop an explicit batch-1 dimension because NMSPRE expects the 2D model head.
    std::vector<int> output_dims = layout_.dims;
    int local_feature_axis = layout_.feature_axis;
    if (output_dims.size() == 3 && output_dims[0] == 1) {
        output_dims.erase(output_dims.begin());
        --local_feature_axis;
    }
    if (output_dims.size() != 2 || local_feature_axis < 0 || local_feature_axis > 1) {
        return false;
    }

    // NMSPRE expects [C,N] when transpose is true, or [N,C] when false.
    if (output_dims[0] == layout_.channels) {
        prenms_need_transpose_ = true;
        prenms_input_height_ = output_dims[0];
        prenms_input_width_ = output_dims[1];
    }
    else if (output_dims[1] == layout_.channels) {
        prenms_need_transpose_ = false;
        prenms_input_height_ = output_dims[0];
        prenms_input_width_ = output_dims[1];
    }
    else {
        return false;
    }

    if (layout_.channels != config_.class_count + 4 || layout_.anchors <= 0 ||
        prenms_input_height_ <= 0 || prenms_input_width_ <= 0) {
        return false;
    }

    releaseRtnmsBuffers();
    resetRtnmsEngines();

    // Build engines first, then allocate buffers sized from their binding dimensions.
    if (!buildPreNmsEngine() || !buildNmsEngine() ||
        !allocateRtnmsBuffers() || !copyRtnmsConstantsToDevice()) {
        releaseRtnmsBuffers();
        resetRtnmsEngines();
        return false;
    }
    return true;
}

/**
 * @brief Build the NMSPRE plugin engine that produces center-size boxes and per-class scores.
 */
bool YoloV8Postprocessor::buildPreNmsEngine()
{
    infer1::Dims input_dims{};
    input_dims.nbDims = 2;
    input_dims.d[0] = prenms_input_height_;
    input_dims.d[1] = prenms_input_width_;

    std::unique_ptr<infer1::IBuilder> builder {infer1::createInferBuilder(rtnms_logger())};
    if (builder == nullptr) {
        return false;
    }
    std::unique_ptr<infer1::INetworkDefinition> network {builder->createNetwork()};
    if (network == nullptr) {
        return false;
    }
    std::unique_ptr<infer1::IBuilderConfig> builder_config {builder->createBuilderConfig()};
    if (builder_config == nullptr) {
        return false;
    }

    infer1::ITensor* input = network->addInput("nmspre_input", infer1::DataType::kFLOAT, input_dims);
    if (input == nullptr) {
        return false;
    }

    NMSPREPlugin plugin;
    plugin.setHeight(prenms_input_height_);
    plugin.setWidth(prenms_input_width_);
    plugin.setParams(config_.score_threshold,
                     config_.nms_threshold,
                     kMaxOutputBoxesPerClass,
                     prenms_need_transpose_,
                     sizeof(float));

    infer1::ITensor* plugin_inputs[] = {input};
    infer1::IPluginV2Layer* plugin_layer = network->addPluginV2(plugin_inputs, 1, plugin);
    if (plugin_layer == nullptr) {
        return false;
    }
    plugin_layer->setPrecision(infer1::DataType::kFLOAT);
    plugin_layer->setOutputType(0, infer1::DataType::kFLOAT);
    plugin_layer->setOutputType(1, infer1::DataType::kFLOAT);

    infer1::ITensor* boxes = plugin_layer->getOutput(0);
    infer1::ITensor* scores = plugin_layer->getOutput(1);
    if (boxes == nullptr || scores == nullptr) {
        return false;
    }
    boxes->setName("box_buf");
    scores->setName("score_buf");
    network->markOutput(*boxes);
    network->markOutput(*scores);

    builder_config->setMaxWorkspaceSize(1024);
    builder_config->setFlag(infer1::BuilderFlag::kBF16);
    prenms_engine_.reset(builder->buildEngineWithConfig(*network, *builder_config));
    if (prenms_engine_ == nullptr) {
        return false;
    }
    prenms_context_.reset(prenms_engine_->createExecutionContext());
    if (prenms_context_ == nullptr) {
        return false;
    }

    prenms_input_index_ = prenms_engine_->getBindingIndex("nmspre_input");
    prenms_boxes_index_ = prenms_engine_->getBindingIndex("box_buf");
    prenms_scores_index_ = prenms_engine_->getBindingIndex("score_buf");
    return prenms_input_index_ >= 0 && prenms_boxes_index_ >= 0 && prenms_scores_index_ >= 0;
}

/**
 * @brief Build the OpenRT NMS engine that deduplicates PreNMS boxes by class scores.
 */
bool YoloV8Postprocessor::buildNmsEngine()
{
    infer1::Dims input_boxes_dims{};
    input_boxes_dims.nbDims = 2;
    input_boxes_dims.d[0] = layout_.anchors;
    input_boxes_dims.d[1] = 4;

    infer1::Dims input_scores_dims{};
    input_scores_dims.nbDims = 2;
    input_scores_dims.d[0] = layout_.anchors;
    input_scores_dims.d[1] = config_.class_count;

    infer1::Dims scalar_dims{};
    scalar_dims.nbDims = 1;
    scalar_dims.d[0] = 1;

    std::unique_ptr<infer1::IBuilder> builder {infer1::createInferBuilder(rtnms_logger())};
    if (builder == nullptr) {
        return false;
    }
    std::unique_ptr<infer1::INetworkDefinition> network {builder->createNetwork()};
    if (network == nullptr) {
        return false;
    }
    std::unique_ptr<infer1::IBuilderConfig> builder_config {builder->createBuilderConfig()};
    if (builder_config == nullptr) {
        return false;
    }

    infer1::ITensor* input_boxes = network->addInput("input_boxes", infer1::DataType::kFLOAT, input_boxes_dims);
    infer1::ITensor* input_scores = network->addInput("input_scores", infer1::DataType::kFLOAT, input_scores_dims);
    infer1::ITensor* input_max_output = network->addInput("input_max_out_per_class", infer1::DataType::kINT32, scalar_dims);
    infer1::ITensor* input_iou_threshold = network->addInput("input_iou_threshold", infer1::DataType::kFLOAT, scalar_dims);
    infer1::ITensor* input_score_threshold = network->addInput("input_score_threshold", infer1::DataType::kFLOAT, scalar_dims);
    if (input_boxes == nullptr || input_scores == nullptr || input_max_output == nullptr ||
        input_iou_threshold == nullptr || input_score_threshold == nullptr) {
        return false;
    }

    infer1::INMSLayer* nms_layer = network->addNMS(*input_boxes, *input_scores, *input_max_output);
    if (nms_layer == nullptr) {
        return false;
    }
    nms_layer->setBoundingBoxFormat(infer1::BoundingBoxFormat::kCENTER_SIZES);
    nms_layer->setInput(3, *input_iou_threshold);
    nms_layer->setInput(4, *input_score_threshold);
    nms_layer->setOutputType(0, infer1::DataType::kINT32);
    nms_layer->setOutputType(1, infer1::DataType::kINT32);
    // YOLOv8 class channels already contain objectness-folded scores.
    nms_layer->setHasBoxConfidence(false);
    nms_layer->setIsAgnostic(config_.agnostic);

    infer1::ITensor* output_indices = nms_layer->getOutput(0);
    infer1::ITensor* output_count = nms_layer->getOutput(1);
    if (output_indices == nullptr || output_count == nullptr) {
        return false;
    }
    output_indices->setName("output_indices");
    output_count->setName("output_num_out_boxes");
    network->markOutput(*output_indices);
    network->markOutput(*output_count);

    builder_config->setMaxWorkspaceSize(1024);
    builder_config->setFlag(infer1::BuilderFlag::kBF16);
    nms_engine_.reset(builder->buildEngineWithConfig(*network, *builder_config));
    if (nms_engine_ == nullptr) {
        return false;
    }
    nms_context_.reset(nms_engine_->createExecutionContext());
    if (nms_context_ == nullptr) {
        return false;
    }

    nms_boxes_index_ = nms_engine_->getBindingIndex("input_boxes");
    nms_scores_index_ = nms_engine_->getBindingIndex("input_scores");
    nms_max_output_index_ = nms_engine_->getBindingIndex("input_max_out_per_class");
    nms_iou_threshold_index_ = nms_engine_->getBindingIndex("input_iou_threshold");
    nms_score_threshold_index_ = nms_engine_->getBindingIndex("input_score_threshold");
    nms_indices_index_ = nms_engine_->getBindingIndex("output_indices");
    nms_count_index_ = nms_engine_->getBindingIndex("output_num_out_boxes");
    if (nms_boxes_index_ < 0 || nms_scores_index_ < 0 || nms_max_output_index_ < 0 ||
        nms_iou_threshold_index_ < 0 || nms_score_threshold_index_ < 0 ||
        nms_indices_index_ < 0 || nms_count_index_ < 0) {
        return false;
    }

    infer1::Dims indices_dims = nms_engine_->getBindingDimensions(nms_indices_index_);
    infer1::Dims count_dims = nms_engine_->getBindingDimensions(nms_count_index_);
    nms_indices_elements_ = dims_volume(indices_dims);
    nms_count_elements_ = dims_volume(count_dims);
    // Most OpenRT NMS builds return [class_id, box_id]; keep support for a leading batch id.
    if (indices_dims.nbDims > 0 && indices_dims.d[indices_dims.nbDims - 1] > 0 &&
        indices_dims.d[indices_dims.nbDims - 1] <= 4) {
        nms_index_tuple_size_ = indices_dims.d[indices_dims.nbDims - 1];
    }
    else {
        nms_index_tuple_size_ = 2;
    }
    return nms_indices_elements_ > 0 && nms_count_elements_ > 0 && nms_index_tuple_size_ >= 2;
}

/**
 * @brief Allocate device inputs/outputs and pinned host staging buffers for RT_NMS.
 */
bool YoloV8Postprocessor::allocateRtnmsBuffers()
{
    if (prenms_engine_ == nullptr || nms_engine_ == nullptr) {
        return false;
    }

    int boxes_elements = dims_volume(prenms_engine_->getBindingDimensions(prenms_boxes_index_));
    int scores_elements = dims_volume(prenms_engine_->getBindingDimensions(prenms_scores_index_));
    if (boxes_elements <= 0 || scores_elements <= 0 ||
        boxes_elements != layout_.anchors * 4 ||
        scores_elements != layout_.anchors * config_.class_count) {
        return false;
    }

    boxes_bytes_ = static_cast<size_t>(boxes_elements) * sizeof(float);
    scores_bytes_ = static_cast<size_t>(scores_elements) * sizeof(float);
    nms_outputs_bytes_ = static_cast<size_t>(nms_indices_elements_ + nms_count_elements_) * sizeof(int32_t);

    // Boxes/scores are copied back in contiguous chunks so result gathering avoids many tiny D2H transfers.
    if (!allocate_device(&boxes_device_, boxes_bytes_, "rtMalloc RT_NMS boxes") ||
        !allocate_device(&scores_device_, scores_bytes_, "rtMalloc RT_NMS scores") ||
        !allocate_device(&max_output_per_class_device_, sizeof(int32_t), "rtMalloc RT_NMS max output") ||
        !allocate_device(&iou_threshold_device_, sizeof(float), "rtMalloc RT_NMS iou threshold") ||
        !allocate_device(&score_threshold_device_, sizeof(float), "rtMalloc RT_NMS score threshold") ||
        !allocate_device(&nms_outputs_device_, nms_outputs_bytes_, "rtMalloc RT_NMS outputs") ||
        !allocate_host_pinned(&boxes_host_, boxes_bytes_, "rtMallocHost RT_NMS boxes") ||
        !allocate_host_pinned(&scores_host_, scores_bytes_, "rtMallocHost RT_NMS scores") ||
        !allocate_host_pinned(&nms_outputs_host_, nms_outputs_bytes_, "rtMallocHost RT_NMS outputs")) {
        return false;
    }

    nms_indices_device_ = nms_outputs_device_;
    nms_count_device_ = static_cast<void*>(static_cast<int32_t*>(nms_outputs_device_) + nms_indices_elements_);
    return true;
}

/**
 * @brief Copy constant NMS parameters to their runtime input bindings.
 */
bool YoloV8Postprocessor::copyRtnmsConstantsToDevice()
{
    int32_t max_output = kMaxOutputBoxesPerClass;
    float iou_threshold = config_.nms_threshold;
    float score_threshold = config_.score_threshold;

    return check_rt(rtMemcpy(max_output_per_class_device_, &max_output, sizeof(max_output), rtMemcpyHostToDevice),
                    "rtMemcpy RT_NMS max output") &&
           check_rt(rtMemcpy(iou_threshold_device_, &iou_threshold, sizeof(iou_threshold), rtMemcpyHostToDevice),
                    "rtMemcpy RT_NMS iou threshold") &&
           check_rt(rtMemcpy(score_threshold_device_, &score_threshold, sizeof(score_threshold), rtMemcpyHostToDevice),
                    "rtMemcpy RT_NMS score threshold");
}

/**
 * @brief Execute NMSPRE + RT_NMS and convert selected indices to clipped source-image detections.
 */
bool YoloV8Postprocessor::runRtnms(const void* yolo_output_device,
                                   const LetterboxInfo& letterbox,
                                   std::vector<Detection>& detections,
                                   rtStream_t stream,
                                   YoloV8PostprocessProfile* profile)
{
    if (stream == nullptr || prenms_context_ == nullptr || nms_context_ == nullptr ||
        boxes_device_ == nullptr || scores_device_ == nullptr || nms_outputs_device_ == nullptr ||
        boxes_host_ == nullptr || scores_host_ == nullptr || nms_outputs_host_ == nullptr) {
        return false;
    }

    auto total_start = ProfileClock::now();
    auto stage_start = total_start;

    std::vector<void*> prenms_bindings(static_cast<size_t>(prenms_engine_->getNbBindings()), nullptr);
    prenms_bindings[static_cast<size_t>(prenms_input_index_)] = const_cast<void*>(yolo_output_device);
    prenms_bindings[static_cast<size_t>(prenms_boxes_index_)] = boxes_device_;
    prenms_bindings[static_cast<size_t>(prenms_scores_index_)] = scores_device_;

    if (!prenms_context_->enqueue(1, prenms_bindings.data(), stream, nullptr)) {
        rtStreamSynchronize(stream);
        return false;
    }

    std::vector<void*> nms_bindings(static_cast<size_t>(nms_engine_->getNbBindings()), nullptr);
    nms_bindings[static_cast<size_t>(nms_boxes_index_)] = boxes_device_;
    nms_bindings[static_cast<size_t>(nms_scores_index_)] = scores_device_;
    nms_bindings[static_cast<size_t>(nms_max_output_index_)] = max_output_per_class_device_;
    nms_bindings[static_cast<size_t>(nms_iou_threshold_index_)] = iou_threshold_device_;
    nms_bindings[static_cast<size_t>(nms_score_threshold_index_)] = score_threshold_device_;
    nms_bindings[static_cast<size_t>(nms_indices_index_)] = nms_indices_device_;
    nms_bindings[static_cast<size_t>(nms_count_index_)] = nms_count_device_;

    if (!nms_context_->enqueue(1, nms_bindings.data(), stream, nullptr)) {
        rtStreamSynchronize(stream);
        return false;
    }

    if (!check_rt(rtMemcpyAsync(nms_outputs_host_,
                                nms_outputs_device_,
                                nms_outputs_bytes_,
                                rtMemcpyDeviceToHost,
                                stream),
                  "rtMemcpyAsync RT_NMS outputs D2H") ||
        !check_rt(rtMemcpyAsync(boxes_host_,
                                boxes_device_,
                                boxes_bytes_,
                                rtMemcpyDeviceToHost,
                                stream),
                  "rtMemcpyAsync RT_NMS boxes D2H") ||
        !check_rt(rtMemcpyAsync(scores_host_,
                                scores_device_,
                                scores_bytes_,
                                rtMemcpyDeviceToHost,
                                stream),
                  "rtMemcpyAsync RT_NMS scores D2H") ||
        !check_rt(rtStreamSynchronize(stream), "rtStreamSynchronize RT_NMS postprocess")) {
        return false;
    }

    if (profile != nullptr) {
        profile->device_to_host_ms = elapsed_ms(stage_start);
        profile->device_to_host_bytes = nms_outputs_bytes_ + boxes_bytes_ + scores_bytes_;
    }

    stage_start = ProfileClock::now();
    const int32_t* indices = static_cast<const int32_t*>(nms_outputs_host_);
    const int32_t* count_ptr = indices + nms_indices_elements_;
    int selected_count = count_ptr[0];
    int max_selected_count = nms_indices_elements_ / nms_index_tuple_size_;
    selected_count = std::max(0, std::min(selected_count, max_selected_count));

    detections.reserve(static_cast<size_t>(selected_count));
    const float* host_boxes = static_cast<const float*>(boxes_host_);
    const float* host_scores = static_cast<const float*>(scores_host_);
    for (int i = 0; i < selected_count; ++i) {
        const int tuple_offset = i * nms_index_tuple_size_;
        int class_id = indices[tuple_offset];
        int box_index = indices[tuple_offset + 1];
        if (nms_index_tuple_size_ >= 3) {
            class_id = indices[tuple_offset + 1];
            box_index = indices[tuple_offset + 2];
        }
        if (class_id < 0 || class_id >= config_.class_count ||
            box_index < 0 || box_index >= layout_.anchors) {
            continue;
        }

        const float* box_values = host_boxes + static_cast<size_t>(box_index) * 4U;
        float score = host_scores[static_cast<size_t>(box_index) * static_cast<size_t>(config_.class_count) +
                                  static_cast<size_t>(class_id)];
        if (score < config_.score_threshold || box_values[2] <= 0.0f || box_values[3] <= 0.0f) {
            continue;
        }

        float cx = box_values[0];
        float cy = box_values[1];
        float width = box_values[2];
        float height = box_values[3];
        float left = ((cx - 0.5f * width) - letterbox.pad_x) / letterbox.scale;
        float top = ((cy - 0.5f * height) - letterbox.pad_y) / letterbox.scale;
        float right = ((cx + 0.5f * width) - letterbox.pad_x) / letterbox.scale;
        float bottom = ((cy + 0.5f * height) - letterbox.pad_y) / letterbox.scale;

        cv::Rect rect = clip_rect_to_image(left, top, right, bottom, letterbox.source_width, letterbox.source_height);
        if (rect.empty()) {
            continue;
        }

        Detection detection;
        detection.class_id = class_id;
        detection.confidence = score;
        detection.box = rect;
        detections.push_back(detection);
    }

    if (profile != nullptr) {
        profile->decode_nms_ms = elapsed_ms(stage_start);
        profile->total_ms = elapsed_ms(total_start);
    }

    if (std::getenv("YOLO_DEBUG_POST") != nullptr) {
        std::cout << "YOLOv8 postprocess debug: mode=rt_nms"
                  << ", channels=" << layout_.channels
                  << ", anchors=" << layout_.anchors
                  << ", transpose=" << prenms_need_transpose_
                  << ", selected=" << selected_count
                  << ", kept=" << detections.size()
                  << std::endl;
        if (std::getenv("YOLO_DEBUG_POST_VERBOSE") != nullptr) {
            for (size_t i = 0; i < detections.size(); ++i) {
                const Detection& det = detections[i];
                std::cout << "  kept[" << i << "]: class=" << det.class_id
                          << ", score=" << det.confidence
                          << ", box=[" << det.box.x
                          << ", " << det.box.y
                          << ", " << det.box.width
                          << ", " << det.box.height << "]"
                          << std::endl;
            }
        }
    }
    return true;
}

/**
 * @brief Copy YOLOv8 output to host memory and use the CPU decode/NMS fallback.
 */
bool YoloV8Postprocessor::runCpuFallback(const void* yolo_output_device,
                                         const LetterboxInfo& letterbox,
                                         std::vector<Detection>& detections,
                                         YoloV8PostprocessProfile* profile)
{
    auto total_start = ProfileClock::now();
    size_t output_bytes = config_.output_elements * sizeof(float);

    auto stage_start = ProfileClock::now();
    if (!check_rt(rtMemcpy(host_output_.data(),
                           yolo_output_device,
                           output_bytes,
                           rtMemcpyDeviceToHost),
                  "rtMemcpy YOLOv8 output D2H")) {
        return false;
    }
    if (profile != nullptr) {
        profile->device_to_host_ms = elapsed_ms(stage_start);
        profile->device_to_host_bytes = output_bytes;
    }

    stage_start = ProfileClock::now();
    decodeDetections(letterbox, detections);
    if (profile != nullptr) {
        profile->decode_nms_ms = elapsed_ms(stage_start);
        profile->total_ms = elapsed_ms(total_start);
    }
    return true;
}

/**
 * @brief Release all buffers used by the RT_NMS path.
 */
void YoloV8Postprocessor::releaseRtnmsBuffers()
{
    if (boxes_device_ != nullptr) {
        rtFree(boxes_device_);
        boxes_device_ = nullptr;
    }
    if (scores_device_ != nullptr) {
        rtFree(scores_device_);
        scores_device_ = nullptr;
    }
    if (boxes_host_ != nullptr) {
        rtFreeHost(boxes_host_);
        boxes_host_ = nullptr;
    }
    if (scores_host_ != nullptr) {
        rtFreeHost(scores_host_);
        scores_host_ = nullptr;
    }
    if (max_output_per_class_device_ != nullptr) {
        rtFree(max_output_per_class_device_);
        max_output_per_class_device_ = nullptr;
    }
    if (iou_threshold_device_ != nullptr) {
        rtFree(iou_threshold_device_);
        iou_threshold_device_ = nullptr;
    }
    if (score_threshold_device_ != nullptr) {
        rtFree(score_threshold_device_);
        score_threshold_device_ = nullptr;
    }
    if (nms_outputs_device_ != nullptr) {
        rtFree(nms_outputs_device_);
        nms_outputs_device_ = nullptr;
    }
    if (nms_outputs_host_ != nullptr) {
        rtFreeHost(nms_outputs_host_);
        nms_outputs_host_ = nullptr;
    }
    nms_indices_device_ = nullptr;
    nms_count_device_ = nullptr;
    boxes_bytes_ = 0;
    scores_bytes_ = 0;
    nms_outputs_bytes_ = 0;
}

/**
 * @brief Drop RT_NMS engines, contexts, and cached binding indices.
 */
void YoloV8Postprocessor::resetRtnmsEngines()
{
    prenms_context_.reset();
    prenms_engine_.reset();
    nms_context_.reset();
    nms_engine_.reset();

    prenms_input_index_ = -1;
    prenms_boxes_index_ = -1;
    prenms_scores_index_ = -1;
    nms_boxes_index_ = -1;
    nms_scores_index_ = -1;
    nms_max_output_index_ = -1;
    nms_iou_threshold_index_ = -1;
    nms_score_threshold_index_ = -1;
    nms_indices_index_ = -1;
    nms_count_index_ = -1;
    nms_index_tuple_size_ = 2;
    nms_indices_elements_ = 0;
    nms_count_elements_ = 0;
}

/**
 * @brief Read one decoded YOLOv8 output value using the resolved output layout.
 */
float YoloV8Postprocessor::valueAt(int channel, int anchor) const
{
    size_t offset = static_cast<size_t>(channel) * layout_.strides[static_cast<size_t>(layout_.feature_axis)];
    int remaining = anchor;

    for (int axis = static_cast<int>(layout_.dims.size()) - 1; axis >= 0; --axis) {
        if (axis == layout_.feature_axis) {
            continue;
        }
        int dim = layout_.dims[static_cast<size_t>(axis)];
        int coord = remaining % dim;
        remaining /= dim;
        offset += static_cast<size_t>(coord) * layout_.strides[static_cast<size_t>(axis)];
    }
    return host_output_[offset];
}

/**
 * @brief Decode YOLOv8 head output on CPU and run class-aware or class-agnostic NMS.
 */
void YoloV8Postprocessor::decodeDetections(const LetterboxInfo& letterbox, std::vector<Detection>& detections) const
{
    std::vector<int> class_ids;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;

    float global_best_score = 0.0f;
    int global_best_anchor = -1;
    float global_best_box[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int threshold_pass_count = 0;

    // YOLOv8 exports boxes as center-format coordinates plus class scores.
    for (int anchor = 0; anchor < layout_.anchors; ++anchor) {
        int best_class_id = 0;
        float best_score = 0.0f;
        for (int channel = 4; channel < layout_.channels; ++channel) {
            float score = valueAt(channel, anchor);
            if (score > best_score) {
                best_score = score;
                best_class_id = channel - 4;
            }
        }
        if (best_score > global_best_score) {
            global_best_score = best_score;
            global_best_anchor = anchor;
            global_best_box[0] = valueAt(0, anchor);
            global_best_box[1] = valueAt(1, anchor);
            global_best_box[2] = valueAt(2, anchor);
            global_best_box[3] = valueAt(3, anchor);
        }
        if (best_score < config_.score_threshold) {
            continue;
        }
        ++threshold_pass_count;

        float cx = valueAt(0, anchor);
        float cy = valueAt(1, anchor);
        float width = valueAt(2, anchor);
        float height = valueAt(3, anchor);
        if (width <= 0.0f || height <= 0.0f) {
            continue;
        }

        // Undo letterbox padding so the final box lands in source-image space.
        float left = ((cx - 0.5f * width) - letterbox.pad_x) / letterbox.scale;
        float top = ((cy - 0.5f * height) - letterbox.pad_y) / letterbox.scale;
        float right = ((cx + 0.5f * width) - letterbox.pad_x) / letterbox.scale;
        float bottom = ((cy + 0.5f * height) - letterbox.pad_y) / letterbox.scale;

        cv::Rect box = clip_rect_to_image(left, top, right, bottom, letterbox.source_width, letterbox.source_height);
        if (box.empty()) {
            continue;
        }

        class_ids.push_back(best_class_id);
        scores.push_back(best_score);
        boxes.push_back(box);
    }

    std::vector<int> keep;
    nms_boxes(boxes,
              scores,
              class_ids,
              config_.agnostic,
              config_.score_threshold,
              config_.nms_threshold,
              keep);

    // Return the same post-NMS Detection structure used by the YOLOv5-style pipeline.
    detections.reserve(keep.size());
    for (int idx : keep) {
        Detection detection;
        detection.class_id = class_ids[static_cast<size_t>(idx)];
        detection.confidence = scores[static_cast<size_t>(idx)];
        detection.box = boxes[static_cast<size_t>(idx)];
        detections.push_back(detection);
    }

    if (std::getenv("YOLO_DEBUG_POST") != nullptr) {
        std::cout << "YOLOv8 postprocess debug: channels=" << layout_.channels
                  << ", anchors=" << layout_.anchors
                  << ", feature_axis=" << layout_.feature_axis
                  << ", max_score=" << global_best_score
                  << ", max_anchor=" << global_best_anchor
                  << ", max_box=[" << global_best_box[0]
                  << ", " << global_best_box[1]
                  << ", " << global_best_box[2]
                  << ", " << global_best_box[3] << "]"
                  << ", score_threshold=" << config_.score_threshold
                  << ", threshold_pass=" << threshold_pass_count
                  << ", valid_boxes=" << boxes.size()
                  << ", kept=" << detections.size()
                  << std::endl;
        if (std::getenv("YOLO_DEBUG_POST_VERBOSE") != nullptr) {
            for (size_t i = 0; i < detections.size(); ++i) {
                const Detection& det = detections[i];
                std::cout << "  kept[" << i << "]: class=" << det.class_id
                          << ", score=" << det.confidence
                          << ", box=[" << det.box.x
                          << ", " << det.box.y
                          << ", " << det.box.width
                          << ", " << det.box.height << "]"
                          << std::endl;
            }
        }
    }
}
