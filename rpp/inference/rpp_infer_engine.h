/**
 * @file rpp_infer_engine.h
 * @brief Reusable OpenRT inference engine wrapper.
 */
#ifndef XDLTEK_SAMPLES_RPP_INFER_ENGINE_H
#define XDLTEK_SAMPLES_RPP_INFER_ENGINE_H

#include "rpp_buffer_manager.h"

#include <cstddef>
#include <memory>
#include <string>

/**
 * @brief Reusable OpenRT inference wrapper that owns the model engine, bindings, and context.
 */
class RppInferEngine
{
public:
    /**
     * @brief Store the ONNX path used during runtime engine initialization.
     * @param onnx_path Path to a YOLO-compatible ONNX model file.
     */
    explicit RppInferEngine(const std::string& onnx_path)
        : onnx_model_path_(onnx_path) {}

    /**
     * @brief Parse ONNX, build the OpenRT engine, allocate bindings, and create execution context.
     */
    bool init();

    /**
     * @brief Return the model input tensor width derived from the ONNX binding.
     */
    int getInputWidth() const { return input_width_; }
    /**
     * @brief Return the model input tensor height derived from the ONNX binding.
     */
    int getInputHeight() const { return input_height_; }
    /**
     * @brief Return the total element count of the model output tensor.
     */
    size_t getOutputSize() const { return output_tensor_size_; }

    /**
     * @brief Return the model output tensor dimensions.
     */
    infer1::Dims getOutputDimensions() const { return output_dimensions_; }
    /**
     * @brief Return the model input tensor data type.
     */
    infer1::DataType getInputDataType() const { return input_data_type_; }
    /**
     * @brief Return the model output tensor data type.
     */
    infer1::DataType getOutputDataType() const { return output_data_type_; }

    /**
     * @brief Return the device buffer bound to the model input tensor.
     */
    void* getInputDeviceBuffer() const;
    /**
     * @brief Return the device buffer bound to the model output tensor.
     */
    void* getOutputDeviceBuffer() const;

    /**
     * @brief Execute the runtime context synchronously with current device bindings.
     */
    bool execute();

private:
    /**
     * @brief Run synchronous inference on the current execution context.
     */
    bool executeContext();

    std::unique_ptr<infer1::IRuntime> runtime_ptr_ {nullptr};
    std::shared_ptr<infer1::IEngine> engine_ptr_ {nullptr};
    std::shared_ptr<samplesCommon::RppBufferManager> buffer_ptr_ {nullptr};
    std::unique_ptr<infer1::IExecutionContext> context_ptr_ {nullptr};

    int input_width_ = 0;
    int input_height_ = 0;
    size_t output_tensor_size_ = 0;
    std::string onnx_model_path_;

    std::string input_name_;
    std::string output_name_;
    infer1::Dims output_dimensions_;
    infer1::DataType input_data_type_ {infer1::DataType::kFLOAT};
    infer1::DataType output_data_type_ {infer1::DataType::kFLOAT};
};

#endif // XDLTEK_SAMPLES_RPP_INFER_ENGINE_H
