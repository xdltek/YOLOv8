/**
 * @file rpp_infer_engine.cpp
 * @brief Reusable OpenRT inference engine wrapper implementation.
 */
#include "rpp_infer_engine.h"

#include "logger.h"
#include "parser_api.h"
#include "sampleCommon.h"

#include <exception>
#include <iostream>

namespace
{
/**
 * @brief Read model input tensor dimensions and infer the spatial width and height.
 * @param dims Runtime binding dimensions.
 * @param width Output spatial width.
 * @param height Output spatial height.
 */
bool parseInputSpatialDims(const infer1::Dims& dims, int& width, int& height)
{
    if (dims.nbDims < 2) {
        return false;
    }

    if (dims.nbDims >= 3 && dims.d[dims.nbDims - 1] == 3) {
        height = dims.d[dims.nbDims - 3];
        width = dims.d[dims.nbDims - 2];
    }
    else {
        height = dims.d[dims.nbDims - 2];
        width = dims.d[dims.nbDims - 1];
    }

    return width > 0 && height > 0;
}

}

/**
 * @brief Build the OpenRT engine from ONNX and prepare device binding buffers.
 */
bool RppInferEngine::init()
{
    std::cout << "[ENGINE  ] Building an in-memory BF16 engine from complete ONNX: "
              << onnx_model_path_ << std::endl;
    // Create the builder, network, and config objects that turn ONNX into an executable engine.
    std::unique_ptr<infer1::IBuilder> builder {infer1::createInferBuilder(sample::gLogger.getLogger())};
    if (builder == nullptr) {
        std::cerr << "Unable to create builder object." << std::endl;
        return false;
    }

    std::unique_ptr<infer1::INetworkDefinition> network {builder->createNetwork()};
    if (network == nullptr) {
        std::cerr << "Unable to create network object." << std::endl;
        return false;
    }

    std::unique_ptr<infer1::IBuilderConfig> config {builder->createBuilderConfig()};
    if (config == nullptr) {
        std::cerr << "Unable to create config object." << std::endl;
        return false;
    }

    // Parse the configured ONNX file into the runtime network definition.
    std::unique_ptr<onnxparser::IParser> parser {onnxparser::createParser(*network, sample::gLogger.getLogger())};
    if (parser == nullptr) {
        std::cerr << "Unable to parse ONNX model file: " << onnx_model_path_ << std::endl;
        return false;
    }
    try {
        if (onnx_parser(onnx_model_path_, builder.get(), network.get(), parser.get()) != 0) {
            return false;
        }
    }
    catch (const std::exception& error) {
        std::cerr << "OpenRT could not import the ONNX model: " << error.what() << std::endl;
        return false;
    }

    // Preserve the complete original ONNX graph and keep the engine in memory
    // for the lifetime of this inference object. Disk serialization is disabled.
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize((size_t)1_GiB);
    config->setFlag(BuilderFlag::kBF16);

    engine_ptr_ = std::shared_ptr<infer1::IEngine>(builder->buildEngineWithConfig(*network, *config));
    if (!engine_ptr_) {
        return false;
    }

    if (config->getFlag(infer1::BuilderFlag::kINT8)) {
        sample::user_visible_log("Since the model includes quantization layers, inference is performed with int8 precision.");
    }

    std::cout << "[ENGINE  ] In-memory BF16 engine is ready." << std::endl;

    // Record binding names, shapes, data types, and element counts for pipeline stages.
    for (int i = 0; i < engine_ptr_->getNbBindings(); i++) {
        if (engine_ptr_->bindingIsInput(i)) {
            input_name_ = engine_ptr_->getBindingName(i);
            infer1::Dims input_dimensions = engine_ptr_->getBindingDimensions(i);
            input_data_type_ = engine_ptr_->getBindingDataType(i);

            if (!parseInputSpatialDims(input_dimensions, input_width_, input_height_)) {
                std::cerr << "Invalid model input dimensions." << std::endl;
                return false;
            }
        }
        else {
            output_name_ = engine_ptr_->getBindingName(i);
            output_dimensions_ = engine_ptr_->getBindingDimensions(i);
            output_data_type_ = engine_ptr_->getBindingDataType(i);
            output_tensor_size_ = samplesCommon::volume(output_dimensions_);
        }
    }

    // Allocate host and device buffers for every engine binding.
    buffer_ptr_ = std::make_shared<samplesCommon::RppBufferManager>(engine_ptr_);
    if (buffer_ptr_ == nullptr) {
        return false;
    }

    // Create the execution context used by execute().
    context_ptr_.reset(engine_ptr_->createExecutionContext());
    return context_ptr_ != nullptr;
}

/**
 * @brief Return the device buffer associated with the model input binding.
 */
void* RppInferEngine::getInputDeviceBuffer() const
{
    return buffer_ptr_ == nullptr ? nullptr : buffer_ptr_->getDeviceBuffer(input_name_);
}

/**
 * @brief Return the device buffer associated with the model output binding.
 */
void* RppInferEngine::getOutputDeviceBuffer() const
{
    return buffer_ptr_ == nullptr ? nullptr : buffer_ptr_->getDeviceBuffer(output_name_);
}

/**
 * @brief Execute one synchronous model inference using current device bindings.
 */
bool RppInferEngine::executeContext()
{
    // Synchronous execution is the inference timing region reported by the demos.
    if (context_ptr_ == nullptr || buffer_ptr_ == nullptr) {
        sample::LOG_ERROR() << "RPP inference engine has not been initialized." << std::endl;
        return false;
    }

    bool ok = context_ptr_->execute(1, buffer_ptr_->getDeviceBindings().data());
    if (!ok) {
        sample::LOG_ERROR() << "do inference failed." << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief Public inference entry point used by the YOLO pipeline.
 */
bool RppInferEngine::execute()
{
    return executeContext();
}
