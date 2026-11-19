/**
 * @file yolo_pipeline.h
 * @brief Reusable YOLOv8 pipeline built from preprocessing, inference, and postprocessing modules.
 */
#ifndef XDLTEK_SAMPLES_YOLOV8_PIPELINE_H
#define XDLTEK_SAMPLES_YOLOV8_PIPELINE_H

#include "detection_types.h"

#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class RppInferEngine;
class RppYoloPreprocessor;
class YoloV8Postprocessor;

/**
 * @brief User-visible options used to initialize a YOLOv8 pipeline instance.
 */
struct YoloV8PipelineOptions
{
    std::string model_path;
    int inference_count = 1;
    YoloV8Config config;
};

/**
 * @brief Stage-level timing summary reported by demos after one measured pipeline run.
 */
struct YoloV8StageTimes
{
    double input_h2d_ms = 0.0;
    size_t input_h2d_bytes = 0;
    double output_d2h_ms = 0.0;
    size_t output_d2h_bytes = 0;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double end_to_end_ms = 0.0;
};

/**
 * @brief Reusable YOLOv8 pipeline that wires preprocessing, inference, and postprocessing.
 */
class YoloV8Pipeline
{
public:
    /**
     * @brief Construct an empty pipeline; call init() before runRgb() or runI420().
     */
    YoloV8Pipeline();
    /**
     * @brief Destroy owned preprocessing, inference, and postprocessing modules.
     */
    ~YoloV8Pipeline();

    /**
     * @brief Disable copying because the pipeline owns runtime resources.
     */
    YoloV8Pipeline(const YoloV8Pipeline&) = delete;
    /**
     * @brief Disable copy assignment because the pipeline owns runtime resources.
     */
    YoloV8Pipeline& operator=(const YoloV8Pipeline&) = delete;

    /**
     * @brief Initialize the model runtime and YOLOv8 postprocess path.
     * @param options Model path, inference loop count, and configurable thresholds.
     * @param class_names Class name table used to derive class count and visualization text.
     */
    bool init(const YoloV8PipelineOptions& options, const std::vector<std::string>& class_names);

    /**
     * @brief Run one BGR/RGB host image through preprocessing, inference, and postprocessing.
     * @param image Host OpenCV image. Non-contiguous images are cloned before preprocessing.
     * @param detections Output detections restored to source-image coordinates.
     * @param times Optional per-stage timing output.
     */
    bool runRgb(const cv::Mat& image,
                std::vector<Detection>& detections,
                YoloV8StageTimes* times = nullptr);

    /**
     * @brief Run one I420 host frame through preprocessing, inference, and postprocessing.
     * @param yuv_data Host buffer packed as Y plane, U plane, then V plane.
     * @param yuv_bytes Number of bytes available in yuv_data.
     * @param width Source frame width; must be even for I420.
     * @param height Source frame height; must be even for I420.
     * @param detections Output detections restored to source-frame coordinates.
     * @param times Optional per-stage timing output.
     */
    bool runI420(const void* yuv_data,
                 size_t yuv_bytes,
                 int width,
                 int height,
                 std::vector<Detection>& detections,
                 YoloV8StageTimes* times = nullptr);

    /**
     * @brief Return the validated runtime configuration used by this pipeline.
     */
    const YoloV8Config& runtimeConfig() const { return config_; }

private:
    /**
     * @brief Run one generic input descriptor through preprocessing, inference, and postprocessing.
     */
    bool runInput(const void* data,
                  size_t bytes,
                  int width,
                  int height,
                  int row_stride_bytes,
                  int channels,
                  bool continuous,
                  int format,
                  std::vector<Detection>& detections,
                  YoloV8StageTimes* times);

    /**
     * @brief Create or reuse preprocessing workspace for a source shape.
     */
    bool ensurePreprocessor(int width, int height);
    /**
     * @brief Initialize YOLOv8 postprocessing after model output dimensions are known.
     */
    bool initPostprocessor(const std::vector<std::string>& class_names);

    std::unique_ptr<RppInferEngine> infer_engine_;
    std::unique_ptr<RppYoloPreprocessor> preprocessor_;
    std::unique_ptr<YoloV8Postprocessor> postprocessor_;

    YoloV8PipelineOptions options_;
    YoloV8Config config_;
    int preprocessor_width_ = 0;
    int preprocessor_height_ = 0;
    bool initialized_ = false;
};

#endif // XDLTEK_SAMPLES_YOLOV8_PIPELINE_H
