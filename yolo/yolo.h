/**
 * @file yolo.h
 * @brief RPP runtime wrapper around an ONNX YOLO model
 *
 * How to use (typical order):
 *   1. Construct `Yolo` with the path to your `.onnx` file.
 *   2. Call `init_engine()` once: parses ONNX, builds the engine, allocates host/device buffers.
 *   3. Build a float NCHW `cv::Mat` blob (see `main.cpp` letterbox helper) matching `getInputWidth()` × `getInputHeight()`.
 *   4. Call `infer(blob, loop_count, output_vector)`; raw floats are written to `output_vector` (layout follows the model output binding).
 *   5. Use `getOutputDimensions()` in post-processing to interpret `[C, N]` or `[1, C, N]` tensors (YOLOv8 head).
 */
#ifndef XDLTEK_SAMPLES_YOLO_H
#define XDLTEK_SAMPLES_YOLO_H

#include "rpp_buffer_manager.h"

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>


class Yolo {
public:
    explicit Yolo(const std::string& onnx_path)
        : onnx_model_path_(onnx_path) {}

    /**
     * @brief Parse ONNX, build the execution engine, and create `RppBufferManager` for I/O bindings.
     * @return True when engine and buffers are initialized successfully.
     */
    bool init_engine();

    /**
     * @brief Copy `processed_image` into the input binding, run inference, copy output to host, fill `output_data`.
     * @param processed_image 4D float blob [1,3,H,W] contiguous (OpenCV `Mat`); H/W must match engine input.
     * @param inference_count Number of timed `execute` calls after one warmup (used for benchmarking).
     * @param output_data Flattened output tensor (row-major); size equals `getOutputSize()` elements.
     */
    bool infer(cv::Mat& processed_image, int inference_count, std::vector<float>& output_data);

    /** Spatial size of the model input (NCHW: height = d[1], width = d[2] for standard exports). */
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }
    /** Element count of the output tensor (`C * N` for a 2D head). */
    size_t getOutputSize() const { return output_tensor_size_; }

    /** Full binding shapes from the engine (use for YOLOv8 decode: channels × anchors). */
    infer1::Dims getInputDimensions() const { return input_dimensions_; }
    infer1::Dims getOutputDimensions() const { return output_dimensions_; }

private:
    std::shared_ptr<infer1::IEngine> engine_ptr_ {nullptr};
    std::shared_ptr<samplesCommon::RppBufferManager> buffer_ptr_ {nullptr};

    int input_index_ = 0;
    int output_index_ = 0;
    int input_width_ = 0;
    int input_height_ = 0;
    size_t input_tensor_size_ = 0;
    size_t output_tensor_size_ = 0;
    std::string onnx_model_path_;

    std::string input_name_;
    std::string output_name_;
    infer1::Dims input_dimensions_;
    infer1::Dims output_dimensions_;
    infer1::DataType input_data_type_;
    infer1::DataType output_data_type_;
};


#endif // XDLTEK_SAMPLES_YOLO_H
