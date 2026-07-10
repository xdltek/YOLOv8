/**
 * @file visualization.h
 * @brief YOLOv8 visualization helpers.
 */
#ifndef XDLTEK_SAMPLES_YOLOV8_VISUALIZATION_H
#define XDLTEK_SAMPLES_YOLOV8_VISUALIZATION_H

#include "detection_types.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

/**
 * @brief Draw retained detections on a host BGR image.
 * @param image Image modified in place with boxes and labels.
 * @param detections Detection records in source-image coordinates.
 * @param class_names Label table used for class id display.
 */
void draw_detections(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& class_names);
/**
 * @brief Save the rendered detection image to disk.
 * @param output_path Destination image path accepted by OpenCV image codecs.
 * @param image Rendered host image.
 */
bool save_detection_image(const std::string& output_path, const cv::Mat& image);

#endif // XDLTEK_SAMPLES_YOLOV8_VISUALIZATION_H
