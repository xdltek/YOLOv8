/**
 * @file rpp_yolov8_postprocessor.cpp
 * @brief YOLOv5-style fused RPP postprocessing with a YOLOv8 transpose.
 */
#include "rpp_yolov8_postprocessor.h"

#include "perf_trace_session.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>

namespace
{
using Clock = std::chrono::high_resolution_clock;
constexpr int kMaxOutputBoxesPerClass = 200;

class PostprocessLogger : public infer1::ILogger
{
public:
    void log(Severity severity, const char* message) override
    {
        if (severity <= Severity::kERROR) {
            std::cerr << "[YOLOv8 RPP postprocess] " << message << std::endl;
        }
    }
};

PostprocessLogger& logger()
{
    static PostprocessLogger instance;
    return instance;
}

double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool check_rt(rtError_t status, const char* operation)
{
    if (status == rtError_t::rtSuccess) {
        return true;
    }
    std::cerr << operation << " failed, rtError=" << static_cast<int>(status) << std::endl;
    return false;
}

int volume(const infer1::Dims& dims)
{
    int result = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0 || result > std::numeric_limits<int>::max() / dims.d[i]) {
            return 0;
        }
        result *= dims.d[i];
    }
    return result;
}

uint16_t float_to_bf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_float(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool allocate_device(void** pointer, size_t bytes, const char* name)
{
    return bytes > 0 && check_rt(rtMalloc(pointer, bytes), name);
}

bool allocate_host(void** pointer, size_t bytes, const char* name)
{
    return bytes > 0 && check_rt(rtHostAlloc(pointer, bytes, 0), name);
}

cv::Rect restore_box(const uint16_t* box, const LetterboxInfo& letterbox)
{
    const float cx = bf16_to_float(box[0]);
    const float cy = bf16_to_float(box[1]);
    const float width = bf16_to_float(box[2]);
    const float height = bf16_to_float(box[3]);
    if (width <= 0.0f || height <= 0.0f) {
        return {};
    }

    float left = ((cx - 0.5f * width) - letterbox.pad_x) / letterbox.scale;
    float top = ((cy - 0.5f * height) - letterbox.pad_y) / letterbox.scale;
    float right = ((cx + 0.5f * width) - letterbox.pad_x) / letterbox.scale;
    float bottom = ((cy + 0.5f * height) - letterbox.pad_y) / letterbox.scale;

    left = std::max(0.0f, std::min(left, static_cast<float>(letterbox.source_width - 1)));
    top = std::max(0.0f, std::min(top, static_cast<float>(letterbox.source_height - 1)));
    right = std::max(0.0f, std::min(right, static_cast<float>(letterbox.source_width - 1)));
    bottom = std::max(0.0f, std::min(bottom, static_cast<float>(letterbox.source_height - 1)));

    const int x0 = static_cast<int>(left);
    const int y0 = static_cast<int>(top);
    const int x1 = static_cast<int>(right);
    const int y1 = static_cast<int>(bottom);
    return {x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)};
}
}

YoloV8Postprocessor::~YoloV8Postprocessor()
{
    releaseBuffers();
    resetEngine();
}

bool YoloV8Postprocessor::init(const YoloV8Config& config)
{
    PERF_SCOPE_CATE("rpp_yolov8_postprocess_init", "postprocess");
    if (config.class_count <= 0 || config.output_elements == 0 ||
        config.score_threshold < 0.0f || config.score_threshold > 1.0f ||
        config.nms_threshold < 0.0f || config.nms_threshold > 1.0f) {
        std::cerr << "Invalid YOLOv8 postprocess configuration." << std::endl;
        return false;
    }

    releaseBuffers();
    resetEngine();
    config_ = config;
    initialized_ = false;

    if (!resolveOutputLayout() || !buildFusedPostprocessEngine() ||
        !allocateBuffers() || !copyConstantsToDevice()) {
        releaseBuffers();
        resetEngine();
        return false;
    }

    initialized_ = true;
    return true;
}

bool YoloV8Postprocessor::resolveOutputLayout()
{
    std::vector<int> dimensions;
    for (int i = 0; i < config_.output_dimensions.nbDims; ++i) {
        const int dimension = config_.output_dimensions.d[i];
        if (dimension > 0 && !(dimension == 1 && dimensions.empty())) {
            dimensions.push_back(dimension);
        }
    }
    if (dimensions.size() != 2) {
        std::cerr << "YOLOv8 requires a static [1,C,N], [C,N], or [N,C] output." << std::endl;
        return false;
    }

    output_channels_ = config_.class_count + 4;
    if (dimensions[0] == output_channels_) {
        input_is_channel_major_ = true;
        output_anchors_ = dimensions[1];
    }
    else if (dimensions[1] == output_channels_) {
        input_is_channel_major_ = false;
        output_anchors_ = dimensions[0];
    }
    else {
        std::cerr << "Unexpected YOLOv8 output shape: expected one axis to equal class_count + 4 ("
                  << output_channels_ << ")." << std::endl;
        return false;
    }

    const size_t expected = static_cast<size_t>(output_channels_) *
                            static_cast<size_t>(output_anchors_);
    if (output_anchors_ <= 0 || expected != config_.output_elements) {
        std::cerr << "YOLOv8 output element count does not match its resolved layout." << std::endl;
        return false;
    }
    return true;
}

bool YoloV8Postprocessor::buildFusedPostprocessEngine()
{
    PERF_SCOPE_CATE("rpp_yolov8_build_fused_postprocess", "postprocess");
    std::unique_ptr<infer1::IBuilder> builder {infer1::createInferBuilder(logger())};
    std::unique_ptr<infer1::INetworkDefinition> network {
        builder == nullptr ? nullptr : builder->createNetwork()};
    std::unique_ptr<infer1::IBuilderConfig> builder_config {
        builder == nullptr ? nullptr : builder->createBuilderConfig()};
    if (builder == nullptr || network == nullptr || builder_config == nullptr) {
        std::cerr << "Failed to create YOLOv8 fused postprocess builder objects." << std::endl;
        return false;
    }

    infer1::Dims raw_dims{};
    raw_dims.nbDims = 2;
    raw_dims.d[0] = input_is_channel_major_ ? output_channels_ : output_anchors_;
    raw_dims.d[1] = input_is_channel_major_ ? output_anchors_ : output_channels_;
    infer1::ITensor* raw = network->addInput("yolov8_output", infer1::DataType::kFLOAT, raw_dims);
    if (raw == nullptr) {
        return false;
    }

    infer1::ITensor* rows = raw;
    if (input_is_channel_major_) {
        infer1::IShuffleLayer* transpose = network->addShuffle(*raw);
        if (transpose == nullptr) {
            std::cerr << "Failed to add YOLOv8 output transpose." << std::endl;
            return false;
        }
        infer1::Permutation permutation{};
        permutation.order[0] = 1;
        permutation.order[1] = 0;
        transpose->setFirstTranspose(permutation);
        rows = transpose->getOutput(0);
    }

    infer1::IIdentityLayer* cast = network->addIdentity(*rows);
    if (cast == nullptr) {
        return false;
    }
    cast->setPrecision(infer1::DataType::kBF);
    cast->setOutputType(0, infer1::DataType::kBF);
    cast->getOutput(0)->setType(infer1::DataType::kBF);

    infer1::Dims start{};
    start.nbDims = 2;
    start.d[0] = 0;
    start.d[1] = 0;
    infer1::Dims stride{};
    stride.nbDims = 2;
    stride.d[0] = 1;
    stride.d[1] = 1;
    infer1::Dims box_size{};
    box_size.nbDims = 2;
    box_size.d[0] = output_anchors_;
    box_size.d[1] = 4;
    infer1::ISliceLayer* boxes = network->addSlice(*cast->getOutput(0), start, box_size, stride);
    if (boxes == nullptr) {
        return false;
    }
    boxes->setPrecision(infer1::DataType::kBF);
    boxes->setOutputType(0, infer1::DataType::kBF);
    boxes->getOutput(0)->setType(infer1::DataType::kBF);
    boxes->getOutput(0)->setName("output_boxes");

    start.d[1] = 4;
    infer1::Dims score_size{};
    score_size.nbDims = 2;
    score_size.d[0] = output_anchors_;
    score_size.d[1] = config_.class_count;
    infer1::ISliceLayer* scores = network->addSlice(*cast->getOutput(0), start, score_size, stride);
    if (scores == nullptr) {
        return false;
    }
    scores->setPrecision(infer1::DataType::kBF);
    scores->setOutputType(0, infer1::DataType::kBF);
    scores->getOutput(0)->setType(infer1::DataType::kBF);
    scores->getOutput(0)->setName("output_scores");

    infer1::Dims scalar{};
    scalar.nbDims = 1;
    scalar.d[0] = 1;
    infer1::ITensor* max_output = network->addInput("max_output", infer1::DataType::kINT32, scalar);
    infer1::ITensor* iou = network->addInput("iou_threshold", infer1::DataType::kBF, scalar);
    infer1::ITensor* score = network->addInput("score_threshold", infer1::DataType::kBF, scalar);
    if (max_output == nullptr || iou == nullptr || score == nullptr) {
        return false;
    }

    // Class-agnostic NMS only needs the best class score for each anchor. Make
    // that reduction explicit so large taxonomies (for example OIV7's 601
    // classes) do not exceed the RPP NMS compiler's score-channel capacity.
    infer1::ITensor* nms_scores = scores->getOutput(0);
    if (config_.agnostic) {
        infer1::IReduceLayer* best_score = network->addReduce(
            *nms_scores, infer1::ReduceOperation::kMAX, 1U << 1U, true);
        if (best_score == nullptr) {
            return false;
        }
        best_score->setPrecision(infer1::DataType::kBF);
        best_score->setOutputType(0, infer1::DataType::kBF);
        best_score->getOutput(0)->setType(infer1::DataType::kBF);
        nms_scores = best_score->getOutput(0);
    }

    infer1::INMSLayer* nms = network->addNMS(*boxes->getOutput(0), *nms_scores, *max_output);
    if (nms == nullptr) {
        return false;
    }
    nms->setBoundingBoxFormat(infer1::BoundingBoxFormat::kCENTER_SIZES);
    nms->setInput(3, *iou);
    nms->setInput(4, *score);
    nms->setPrecision(infer1::DataType::kBF);
    nms->setOutputType(0, infer1::DataType::kINT32);
    nms->setOutputType(1, infer1::DataType::kINT32);
    nms->setHasBoxConfidence(false);
    nms->setIsAgnostic(false);
    nms->getOutput(0)->setName("output_indices");
    nms->getOutput(1)->setName("output_count");

    network->markOutput(*boxes->getOutput(0));
    network->markOutput(*scores->getOutput(0));
    network->markOutput(*nms->getOutput(0));
    network->markOutput(*nms->getOutput(1));

    builder->setMaxBatchSize(1);
    builder_config->setMaxWorkspaceSize(static_cast<size_t>(16) << 20);
    builder_config->setFlag(infer1::BuilderFlag::kBF16);
    fused_engine_.reset(builder->buildEngineWithConfig(*network, *builder_config));
    fused_context_.reset(fused_engine_ == nullptr ? nullptr : fused_engine_->createExecutionContext());
    if (fused_engine_ == nullptr || fused_context_ == nullptr) {
        std::cerr << "Failed to build YOLOv8 fused RPP postprocess engine." << std::endl;
        return false;
    }

    input_index_ = fused_engine_->getBindingIndex("yolov8_output");
    boxes_index_ = fused_engine_->getBindingIndex("output_boxes");
    scores_index_ = fused_engine_->getBindingIndex("output_scores");
    max_output_index_ = fused_engine_->getBindingIndex("max_output");
    iou_threshold_index_ = fused_engine_->getBindingIndex("iou_threshold");
    score_threshold_index_ = fused_engine_->getBindingIndex("score_threshold");
    indices_index_ = fused_engine_->getBindingIndex("output_indices");
    count_index_ = fused_engine_->getBindingIndex("output_count");
    if (input_index_ < 0 || boxes_index_ < 0 || scores_index_ < 0 ||
        max_output_index_ < 0 || iou_threshold_index_ < 0 || score_threshold_index_ < 0 ||
        indices_index_ < 0 || count_index_ < 0) {
        return false;
    }

    const infer1::Dims indices_dims = fused_engine_->getBindingDimensions(indices_index_);
    indices_elements_ = volume(indices_dims);
    count_elements_ = volume(fused_engine_->getBindingDimensions(count_index_));
    if (indices_dims.nbDims > 0) {
        const int tuple = indices_dims.d[indices_dims.nbDims - 1];
        if (tuple == 2 || tuple == 3) {
            index_tuple_size_ = tuple;
        }
    }
    return indices_elements_ > 0 && count_elements_ > 0;
}

bool YoloV8Postprocessor::allocateBuffers()
{
    boxes_bytes_ = static_cast<size_t>(output_anchors_) * 4U * sizeof(uint16_t);
    scores_bytes_ = static_cast<size_t>(output_anchors_) *
                    static_cast<size_t>(config_.class_count) * sizeof(uint16_t);
    const size_t indices_bytes = static_cast<size_t>(indices_elements_) * sizeof(int32_t);
    nms_outputs_bytes_ = indices_bytes + static_cast<size_t>(count_elements_) * sizeof(int32_t);

    if (!allocate_device(&boxes_device_, boxes_bytes_, "rtMalloc YOLOv8 boxes") ||
        !allocate_device(&scores_device_, scores_bytes_, "rtMalloc YOLOv8 scores") ||
        !allocate_device(&max_output_device_, sizeof(int32_t), "rtMalloc YOLOv8 max output") ||
        !allocate_device(&iou_threshold_device_, sizeof(uint16_t), "rtMalloc YOLOv8 IoU") ||
        !allocate_device(&score_threshold_device_, sizeof(uint16_t), "rtMalloc YOLOv8 score threshold") ||
        !allocate_device(&nms_outputs_device_, nms_outputs_bytes_, "rtMalloc YOLOv8 NMS outputs") ||
        !allocate_host(&boxes_host_, boxes_bytes_, "rtHostAlloc YOLOv8 boxes") ||
        !allocate_host(&scores_host_, scores_bytes_, "rtHostAlloc YOLOv8 scores") ||
        !allocate_host(&nms_outputs_host_, nms_outputs_bytes_, "rtHostAlloc YOLOv8 NMS outputs")) {
        return false;
    }
    indices_device_ = nms_outputs_device_;
    count_device_ = static_cast<unsigned char*>(nms_outputs_device_) + indices_bytes;
    return true;
}

bool YoloV8Postprocessor::copyConstantsToDevice()
{
    const int32_t max_output = kMaxOutputBoxesPerClass;
    const uint16_t iou = float_to_bf16(config_.nms_threshold);
    const uint16_t score = float_to_bf16(config_.score_threshold);
    return check_rt(rtMemcpy(max_output_device_, &max_output, sizeof(max_output), rtMemcpyHostToDevice),
                    "rtMemcpy YOLOv8 max output") &&
           check_rt(rtMemcpy(iou_threshold_device_, &iou, sizeof(iou), rtMemcpyHostToDevice),
                    "rtMemcpy YOLOv8 IoU") &&
           check_rt(rtMemcpy(score_threshold_device_, &score, sizeof(score), rtMemcpyHostToDevice),
                    "rtMemcpy YOLOv8 score threshold");
}

bool YoloV8Postprocessor::run(const void* yolo_output_device,
                              const LetterboxInfo& letterbox,
                              std::vector<Detection>& detections,
                              rtStream_t stream,
                              YoloV8PostprocessProfile* profile)
{
    PERF_SCOPE_CATE("rpp_yolov8_postprocess_run", "postprocess");
    if (profile != nullptr) {
        *profile = {};
    }
    detections.clear();
    if (!initialized_ || yolo_output_device == nullptr || letterbox.scale <= 0.0f ||
        letterbox.source_width <= 0 || letterbox.source_height <= 0) {
        std::cerr << "YOLOv8 RPP postprocessor is not ready or letterbox metadata is invalid." << std::endl;
        return false;
    }

    const auto total_start = Clock::now();
    rtStream_t run_stream = stream;
    const bool owns_stream = run_stream == nullptr;
    if (owns_stream && !check_rt(rtStreamCreate(&run_stream), "rtStreamCreate YOLOv8 postprocess")) {
        return false;
    }

    std::vector<void*> bindings(static_cast<size_t>(fused_engine_->getNbBindings()), nullptr);
    bindings[static_cast<size_t>(input_index_)] = const_cast<void*>(yolo_output_device);
    bindings[static_cast<size_t>(boxes_index_)] = boxes_device_;
    bindings[static_cast<size_t>(scores_index_)] = scores_device_;
    bindings[static_cast<size_t>(max_output_index_)] = max_output_device_;
    bindings[static_cast<size_t>(iou_threshold_index_)] = iou_threshold_device_;
    bindings[static_cast<size_t>(score_threshold_index_)] = score_threshold_device_;
    bindings[static_cast<size_t>(indices_index_)] = indices_device_;
    bindings[static_cast<size_t>(count_index_)] = count_device_;

    auto stage_start = Clock::now();
    if (!fused_context_->enqueue(1, bindings.data(), run_stream, nullptr) ||
        !check_rt(rtStreamSynchronize(run_stream), "rtStreamSynchronize YOLOv8 fused postprocess")) {
        if (owns_stream) {
            rtStreamDestroy(run_stream);
        }
        return false;
    }
    if (profile != nullptr) {
        profile->transpose_nms_ms = elapsed_ms(stage_start);
    }

    stage_start = Clock::now();
    const bool copied =
        check_rt(rtMemcpyAsync(nms_outputs_host_, nms_outputs_device_, nms_outputs_bytes_,
                               rtMemcpyDeviceToHost, run_stream), "rtMemcpyAsync YOLOv8 NMS outputs") &&
        check_rt(rtMemcpyAsync(boxes_host_, boxes_device_, boxes_bytes_,
                               rtMemcpyDeviceToHost, run_stream), "rtMemcpyAsync YOLOv8 boxes") &&
        check_rt(rtMemcpyAsync(scores_host_, scores_device_, scores_bytes_,
                               rtMemcpyDeviceToHost, run_stream), "rtMemcpyAsync YOLOv8 scores") &&
        check_rt(rtStreamSynchronize(run_stream), "rtStreamSynchronize YOLOv8 D2H");
    if (!copied) {
        if (owns_stream) {
            rtStreamDestroy(run_stream);
        }
        return false;
    }
    if (profile != nullptr) {
        profile->device_to_host_ms = elapsed_ms(stage_start);
        profile->device_to_host_bytes = nms_outputs_bytes_ + boxes_bytes_ + scores_bytes_;
    }

    const int32_t* indices = static_cast<const int32_t*>(nms_outputs_host_);
    const int32_t* count = reinterpret_cast<const int32_t*>(
        static_cast<const unsigned char*>(nms_outputs_host_) +
        static_cast<size_t>(indices_elements_) * sizeof(int32_t));
    const int max_selected = indices_elements_ / index_tuple_size_;
    const int selected = std::max(0, std::min(count[0], max_selected));
    stage_start = Clock::now();
    const bool parsed = parseCompactDetections(indices, selected, letterbox, detections);
    if (profile != nullptr) {
        profile->decode_nms_ms = elapsed_ms(stage_start);
        profile->total_ms = elapsed_ms(total_start);
    }

    if (owns_stream) {
        check_rt(rtStreamDestroy(run_stream), "rtStreamDestroy YOLOv8 postprocess");
    }
    return parsed;
}

bool YoloV8Postprocessor::parseCompactDetections(const int32_t* indices,
                                                  int selected_count,
                                                  const LetterboxInfo& letterbox,
                                                  std::vector<Detection>& detections) const
{
    const uint16_t* boxes = static_cast<const uint16_t*>(boxes_host_);
    const uint16_t* scores = static_cast<const uint16_t*>(scores_host_);
    detections.reserve(static_cast<size_t>(selected_count));
    for (int i = 0; i < selected_count; ++i) {
        const size_t tuple = static_cast<size_t>(i) * static_cast<size_t>(index_tuple_size_);
        int class_id = indices[tuple + static_cast<size_t>(index_tuple_size_ - 2)];
        const int box_index = indices[tuple + static_cast<size_t>(index_tuple_size_ - 1)];
        if (box_index < 0 || box_index >= output_anchors_) {
            continue;
        }

        if (config_.agnostic) {
            const uint16_t* box_scores = scores +
                static_cast<size_t>(box_index) * static_cast<size_t>(config_.class_count);
            class_id = 0;
            float best_score = bf16_to_float(box_scores[0]);
            for (int candidate = 1; candidate < config_.class_count; ++candidate) {
                const float candidate_score = bf16_to_float(box_scores[candidate]);
                if (candidate_score > best_score) {
                    best_score = candidate_score;
                    class_id = candidate;
                }
            }
        }
        if (class_id < 0 || class_id >= config_.class_count) {
            continue;
        }

        const float confidence = bf16_to_float(
            scores[static_cast<size_t>(box_index) * static_cast<size_t>(config_.class_count) +
                   static_cast<size_t>(class_id)]);
        if (confidence < config_.score_threshold) {
            continue;
        }
        const cv::Rect box = restore_box(boxes + static_cast<size_t>(box_index) * 4U, letterbox);
        if (box.empty()) {
            continue;
        }
        detections.push_back({class_id, confidence, box});
    }
    return true;
}

void YoloV8Postprocessor::releaseBuffers()
{
    void** device_buffers[] = {&boxes_device_, &scores_device_, &max_output_device_,
                               &iou_threshold_device_, &score_threshold_device_,
                               &nms_outputs_device_};
    for (void** buffer : device_buffers) {
        if (*buffer != nullptr) {
            rtFree(*buffer);
            *buffer = nullptr;
        }
    }
    void** host_buffers[] = {&boxes_host_, &scores_host_, &nms_outputs_host_};
    for (void** buffer : host_buffers) {
        if (*buffer != nullptr) {
            rtFreeHost(*buffer);
            *buffer = nullptr;
        }
    }
    indices_device_ = nullptr;
    count_device_ = nullptr;
    boxes_bytes_ = 0;
    scores_bytes_ = 0;
    nms_outputs_bytes_ = 0;
}

void YoloV8Postprocessor::resetEngine()
{
    fused_context_.reset();
    fused_engine_.reset();
    input_index_ = boxes_index_ = scores_index_ = -1;
    max_output_index_ = iou_threshold_index_ = score_threshold_index_ = -1;
    indices_index_ = count_index_ = -1;
    index_tuple_size_ = 2;
    indices_elements_ = count_elements_ = 0;
}
