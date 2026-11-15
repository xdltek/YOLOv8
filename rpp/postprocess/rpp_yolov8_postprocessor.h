/**
 * @file rpp_yolov8_postprocessor.h
 * @brief Fused RPP transpose, slice, and NMS postprocessor for YOLOv8.
 */
#ifndef XDLTEK_SAMPLES_RPP_YOLOV8_POSTPROCESSOR_H
#define XDLTEK_SAMPLES_RPP_YOLOV8_POSTPROCESSOR_H

#include "detection_types.h"

#include <Infer.h>
#include <rpp_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct YoloV8PostprocessProfile
{
    double transpose_nms_ms = 0.0;
    double device_to_host_ms = 0.0;
    size_t device_to_host_bytes = 0;
    double decode_nms_ms = 0.0;
    double total_ms = 0.0;
};

/**
 * @brief YOLOv8 postprocessor implemented as one RPP runtime graph.
 *
 * A standard YOLOv8 detection head is [1, 4 + classes, anchors]. The graph
 * transposes it to [anchors, 4 + classes], then follows the same box/score
 * slicing and NMS flow used by the YOLOv5 application. YOLOv8 class scores
 * do not have a separate objectness column.
 */
class YoloV8Postprocessor
{
public:
    ~YoloV8Postprocessor();

    YoloV8Postprocessor() = default;
    YoloV8Postprocessor(const YoloV8Postprocessor&) = delete;
    YoloV8Postprocessor& operator=(const YoloV8Postprocessor&) = delete;

    bool init(const YoloV8Config& config);
    bool run(const void* yolo_output_device,
             const LetterboxInfo& letterbox,
             std::vector<Detection>& detections,
             rtStream_t stream = nullptr,
             YoloV8PostprocessProfile* profile = nullptr);

private:
    bool resolveOutputLayout();
    bool buildFusedPostprocessEngine();
    bool allocateBuffers();
    bool copyConstantsToDevice();
    bool parseCompactDetections(const int32_t* indices,
                                int selected_count,
                                const LetterboxInfo& letterbox,
                                std::vector<Detection>& detections) const;
    void releaseBuffers();
    void resetEngine();

    YoloV8Config config_;
    int output_channels_ = 0;
    int output_anchors_ = 0;
    bool input_is_channel_major_ = true;
    bool initialized_ = false;

    std::unique_ptr<infer1::IEngine> fused_engine_;
    std::unique_ptr<infer1::IExecutionContext> fused_context_;
    int input_index_ = -1;
    int boxes_index_ = -1;
    int scores_index_ = -1;
    int max_output_index_ = -1;
    int iou_threshold_index_ = -1;
    int score_threshold_index_ = -1;
    int indices_index_ = -1;
    int count_index_ = -1;
    int index_tuple_size_ = 2;
    int indices_elements_ = 0;
    int count_elements_ = 0;

    void* boxes_device_ = nullptr;
    void* scores_device_ = nullptr;
    void* max_output_device_ = nullptr;
    void* iou_threshold_device_ = nullptr;
    void* score_threshold_device_ = nullptr;
    void* nms_outputs_device_ = nullptr;
    void* indices_device_ = nullptr;
    void* count_device_ = nullptr;
    void* boxes_host_ = nullptr;
    void* scores_host_ = nullptr;
    void* nms_outputs_host_ = nullptr;
    size_t boxes_bytes_ = 0;
    size_t scores_bytes_ = 0;
    size_t nms_outputs_bytes_ = 0;
};

#endif
