/**
 * @file labels.cpp
 * @brief YOLOv8 class label helpers.
 */
#include "labels.h"

/**
 * @brief MS COCO 80 class names in YOLOv8 default order.
 */
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

/**
 * @brief Return a display label for a detection class id.
 */
std::string detection_class_label(int class_id, const std::vector<std::string>& names)
{
    if (class_id >= 0 && class_id < static_cast<int>(names.size())) {
        return names[static_cast<size_t>(class_id)];
    }
    return "class_" + std::to_string(class_id);
}
