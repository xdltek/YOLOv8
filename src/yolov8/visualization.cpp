/**
 * @file visualization.cpp
 * @brief YOLOv8 visualization helpers.
 */
#include "visualization.h"

#include "labels.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace
{
/**
 * @brief Return the fixed palette used when drawing detection boxes.
 */
const std::vector<cv::Scalar>& detection_colors()
{
    static const std::vector<cv::Scalar> kColors = {
        cv::Scalar(255, 255, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 0, 0),
    };
    return kColors;
}
}

/**
 * @brief Draw detection boxes and labels onto an image.
 */
void draw_detections(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& class_names)
{
    const auto& colors = detection_colors();

    for (const auto& detection : detections) {
        cv::Rect box = detection.box & cv::Rect(0, 0, image.cols, image.rows);
        if (box.empty()) {
            continue;
        }

        size_t color_index = detection.class_id >= 0 ? static_cast<size_t>(detection.class_id) % colors.size() : 0;
        const auto color = colors[color_index];
        cv::rectangle(image, box, color, 3);

        int label_top = std::max(0, box.y - 20);
        cv::rectangle(image,
                      cv::Point(box.x, label_top),
                      cv::Point(std::min(image.cols - 1, box.x + box.width), box.y),
                      color,
                      cv::FILLED);
        std::string label = detection_class_label(detection.class_id, class_names);
        cv::putText(image,
                    label,
                    cv::Point(box.x, std::max(12, box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 0, 0));
    }
}

/**
 * @brief Save a visualization image to disk using OpenCV image codecs.
 */
bool save_detection_image(const std::string& output_path, const cv::Mat& image)
{
    return cv::imwrite(output_path, image);
}
