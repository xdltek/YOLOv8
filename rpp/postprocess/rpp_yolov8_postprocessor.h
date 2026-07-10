/**
 * @file rpp_yolov8_postprocessor.h
 * @brief YOLOv8 output copy, decode, and NMS module.
 */
#ifndef XDLTEK_SAMPLES_RPP_YOLOV8_POSTPROCESSOR_H
#define XDLTEK_SAMPLES_RPP_YOLOV8_POSTPROCESSOR_H

#include "detection_types.h"

#include <rpp_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

/**
 * @brief Timing and transfer counters reported by YOLOv8 postprocessing.
 */
struct YoloV8PostprocessProfile
{
    double device_to_host_ms = 0.0;
    size_t device_to_host_bytes = 0;
    double decode_nms_ms = 0.0;
    double total_ms = 0.0;
};

/**
 * @brief YOLOv8 RT_NMS postprocessor with CPU decode/NMS fallback.
 */
class YoloV8Postprocessor
{
public:
    /**
     * @brief Release RT_NMS engines and host/device buffers owned by this postprocessor.
     */
    ~YoloV8Postprocessor();

    /**
     * @brief Validate YOLOv8 output layout and initialize RT_NMS resources when supported.
     * @param config Runtime model output shape, class count, and threshold configuration.
     */
    bool init(const YoloV8Config& config);

    /**
     * @brief Postprocess one device output tensor into final source-image detections.
     * @param yolo_output_device Device pointer to the raw YOLOv8 output binding.
     * @param letterbox Preprocessing geometry used to restore source-image coordinates.
     * @param detections Output NMS-filtered detections.
     * @param stream Optional external stream for RT_NMS execution.
     * @param profile Optional transfer and timing counters.
     */
    bool run(const void* yolo_output_device,
             const LetterboxInfo& letterbox,
             std::vector<Detection>& detections,
             rtStream_t stream = nullptr,
             YoloV8PostprocessProfile* profile = nullptr);

private:
    /**
     * @brief Resolved compact view of the raw YOLOv8 output tensor.
     */
    struct OutputLayout
    {
        std::vector<int> dims;
        std::vector<size_t> strides;
        int feature_axis = -1;
        int channels = 0;
        int anchors = 0;
    };

    /**
     * @brief Resolve supported YOLOv8 output shapes into channel and anchor axes.
     */
    bool resolveOutputLayout();
    /**
     * @brief Initialize the RT_NMS fast path using the already resolved output layout.
     */
    bool initRtnms();
    /**
     * @brief Build the NMSPRE plugin engine that splits YOLOv8 output into boxes and scores.
     */
    bool buildPreNmsEngine();
    /**
     * @brief Build the OpenRT NMS engine that consumes PreNMS boxes and scores.
     */
    bool buildNmsEngine();
    /**
     * @brief Allocate persistent host and device buffers used by RT_NMS.
     */
    bool allocateRtnmsBuffers();
    /**
     * @brief Copy NMS threshold constants into device buffers.
     */
    bool copyRtnmsConstantsToDevice();
    /**
     * @brief Execute the RT_NMS fast path and convert selected indices into Detection records.
     */
    bool runRtnms(const void* yolo_output_device,
                  const LetterboxInfo& letterbox,
                  std::vector<Detection>& detections,
                  rtStream_t stream,
                  YoloV8PostprocessProfile* profile);
    /**
     * @brief Copy the full YOLOv8 output to host and run CPU decode/NMS fallback.
     */
    bool runCpuFallback(const void* yolo_output_device,
                        const LetterboxInfo& letterbox,
                        std::vector<Detection>& detections,
                        YoloV8PostprocessProfile* profile);
    /**
     * @brief Free RT_NMS runtime buffers.
     */
    void releaseRtnmsBuffers();
    /**
     * @brief Reset RT_NMS engine/context owners and binding indices.
     */
    void resetRtnmsEngines();
    /**
     * @brief Read one channel/anchor scalar from the host fallback output buffer.
     */
    float valueAt(int channel, int anchor) const;
    /**
     * @brief Decode full host YOLOv8 output and apply CPU NMS fallback.
     */
    void decodeDetections(const LetterboxInfo& letterbox, std::vector<Detection>& detections) const;

    YoloV8Config config_;
    OutputLayout layout_;
    std::vector<float> host_output_;
    bool initialized_ = false;
    bool rtnms_initialized_ = false;

    std::unique_ptr<infer1::IEngine> prenms_engine_;
    std::unique_ptr<infer1::IExecutionContext> prenms_context_;
    int prenms_input_index_ = -1;
    int prenms_boxes_index_ = -1;
    int prenms_scores_index_ = -1;
    bool prenms_need_transpose_ = false;
    int prenms_input_height_ = 0;
    int prenms_input_width_ = 0;

    std::unique_ptr<infer1::IEngine> nms_engine_;
    std::unique_ptr<infer1::IExecutionContext> nms_context_;
    int nms_boxes_index_ = -1;
    int nms_scores_index_ = -1;
    int nms_max_output_index_ = -1;
    int nms_iou_threshold_index_ = -1;
    int nms_score_threshold_index_ = -1;
    int nms_indices_index_ = -1;
    int nms_count_index_ = -1;
    int nms_index_tuple_size_ = 2;
    int nms_indices_elements_ = 0;
    int nms_count_elements_ = 0;

    void* boxes_device_ = nullptr;
    void* scores_device_ = nullptr;
    void* boxes_host_ = nullptr;
    void* scores_host_ = nullptr;
    void* max_output_per_class_device_ = nullptr;
    void* iou_threshold_device_ = nullptr;
    void* score_threshold_device_ = nullptr;
    void* nms_outputs_device_ = nullptr;
    void* nms_indices_device_ = nullptr;
    void* nms_count_device_ = nullptr;
    void* nms_outputs_host_ = nullptr;
    size_t boxes_bytes_ = 0;
    size_t scores_bytes_ = 0;
    size_t nms_outputs_bytes_ = 0;
};

#endif // XDLTEK_SAMPLES_RPP_YOLOV8_POSTPROCESSOR_H
