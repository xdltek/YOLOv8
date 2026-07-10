/**
 * @file detection_types.h
 * @brief Shared YOLOv8 detection data structures and pipeline configuration.
 */
#ifndef XDLTEK_SAMPLES_YOLOV8_DETECTION_TYPES_H
#define XDLTEK_SAMPLES_YOLOV8_DETECTION_TYPES_H

#include <Infer.h>
#include <opencv2/core.hpp>

/**
 * @brief One YOLOv8 detection restored to original image coordinates.
 */
struct Detection
{
    int class_id = -1;
    float confidence = 0.0f;
    cv::Rect box;
};

/**
 * @brief Geometry metadata shared by preprocessing and postprocessing.
 */
struct LetterboxInfo
{
    int source_width = 0;
    int source_height = 0;
    int model_width = 0;
    int model_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

/**
 * @brief Runtime parameters for a standard Ultralytics YOLOv8 detection model.
 */
struct YoloV8Config
{
    int input_width = 640;
    int input_height = 640;
    float score_threshold = 0.25f;
    float nms_threshold = 0.45f;
    bool agnostic = true;
    int class_count = 80;
    infer1::Dims output_dimensions{};
    size_t output_elements = 0;
};

#endif // XDLTEK_SAMPLES_YOLOV8_DETECTION_TYPES_H
