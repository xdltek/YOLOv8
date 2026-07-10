/**
 * @file labels.h
 * @brief YOLOv8 class label helpers.
 */
#ifndef XDLTEK_SAMPLES_YOLOV8_LABELS_H
#define XDLTEK_SAMPLES_YOLOV8_LABELS_H

#include <string>
#include <vector>

/**
 * @brief Return the COCO 80-class label table used by standard YOLOv8 models.
 */
const std::vector<std::string>& coco80_class_labels();

/**
 * @brief Resolve a class id to display text for rendered detections.
 */
std::string detection_class_label(int class_id, const std::vector<std::string>& names);

#endif // XDLTEK_SAMPLES_YOLOV8_LABELS_H
