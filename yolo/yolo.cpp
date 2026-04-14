/**
 * @file yolo.cpp
 * @brief Implements `Yolo::init_engine` and `Yolo::infer` for RppRT
 *
 * Pipeline summary:
 *   init_engine: IBuilder → INetworkDefinition → onnx_parser → IBuilderConfig (BF16, workspace) → IEngine → RppBufferManager.
 *   infer:       flatten Mat → memcpy to host input → copyInputToDevice → execute (warmup + timed loop) → copyOutputToHost → float vector.
 */
#include "yolo.h"
#include "parser_api.h"
#include "logger.h"
#include "sampleCommon.h"

#include <iostream>
#include <stdexcept>

bool Yolo::init_engine() {

    // --- Build phase: network + config, then compile to IEngine (batch size fixed to 1 for this demo). ---
    std::unique_ptr<infer1::IBuilder> builder {infer1::createInferBuilder(sample::gLogger.getLogger())};
    if (builder == nullptr)
    {
        std::cerr << "Unable to create builder object." << std::endl;
        return false;
    }

    std::unique_ptr<infer1::INetworkDefinition> network { builder->createNetwork() };
    if (network == nullptr)
    {
        std::cerr << "Unable to create network object." << std::endl;
        return false;
    }

    std::unique_ptr<infer1::IBuilderConfig> config {builder->createBuilderConfig()};
    if (config == nullptr)
    {
        std::cerr << "Unable to create config object." << std::endl;
        return false;
    }

    std::unique_ptr<onnxparser::IParser> parser { onnxparser::createParser(*network, sample::gLogger.getLogger()) };
    if (parser == nullptr)
    {
        std::cerr << "Unable to parse ONNX model file: " << onnx_model_path_ << std::endl;
        return false;
    }
    if (onnx_parser(onnx_model_path_, builder.get(), network.get(), parser.get()) != 0) {
        return false;
    }

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize((size_t) 1_GiB);
    config->setFlag(BuilderFlag::kBF16);

    engine_ptr_ = std::shared_ptr<infer1::IEngine>(builder->buildEngineWithConfig(*network, *config));
    if (!engine_ptr_.get()) {
        return false;
    }

    auto has_flag = config->getFlag(infer1::BuilderFlag::kINT8);
    if (has_flag) {
        sample::user_visible_log("Since the model includes quantization layers, inference is performed with int8 precision.");
    }

    // --- Cache binding names and tensor metadata (single input / single output assumed). ---
    for (int i =0; i < engine_ptr_->getNbBindings(); i++) {
        if (engine_ptr_->bindingIsInput(i)) {
            input_name_ = engine_ptr_->getBindingName(i);
            input_dimensions_ = engine_ptr_->getBindingDimensions(i);
            input_data_type_ = engine_ptr_->getBindingDataType(i);

            input_height_ = input_dimensions_.d[1];
            input_width_ = input_dimensions_.d[2];

            input_tensor_size_ = samplesCommon::volume(input_dimensions_);
        }
        else {
            output_name_ = engine_ptr_->getBindingName(i);
            output_dimensions_ = engine_ptr_->getBindingDimensions(i);
            output_data_type_ = engine_ptr_->getBindingDataType(i);

            output_tensor_size_ = samplesCommon::volume(output_dimensions_);
        }
    }
    buffer_ptr_ = std::make_shared<samplesCommon::RppBufferManager>(engine_ptr_);
    if (buffer_ptr_ == nullptr) {
        return false;
    }
    return true;
}

bool Yolo::infer(cv::Mat& processed_image, int inference_count, std::vector<float>& output_data) {

    // Host-side input: contiguous float buffer sized to the engine input element count × element size.
    std::vector<float> input_tensor_values;
    input_tensor_values.assign((float*)processed_image.datastart, (float*)processed_image.dataend);

    size_t input_data_size = input_tensor_size_ * samplesCommon::getElementSize(input_data_type_);
    memcpy(buffer_ptr_->getHostBuffer(input_name_), input_tensor_values.data(), input_data_size);

    std::unique_ptr<infer1::IExecutionContext> context {engine_ptr_->createExecutionContext()};
    buffer_ptr_->copyInputToDevice();

    samplesCommon::PreciseCpuTimer infer_timer;

    // Warmup execute: not included in the timed loop below (avoids first-kernel skew in benchmarks).
    bool ok = context->execute(1, buffer_ptr_->getDeviceBindings().data());
    if (!ok)
    {
        sample::LOG_ERROR() << "do inference failed." << std::endl;
        return false;
    }

    infer_timer.reset();
    infer_timer.start();

    for (int i = 0; i < inference_count; i++)
    {
        ok = context->execute(1, buffer_ptr_->getDeviceBindings().data());
        if (!ok)
        {
            throw std::runtime_error("execute failed.");
        }
    }

    infer_timer.stop();
    float infer_cost = infer_timer.milliseconds();

    if (inference_count == 1) {
        sample::user_visible_stream_log("inference takes: ", infer_cost, "  milliseconds, frames per second: ", int(1000.0f / infer_cost));
    }
    else {
        float average_cost = infer_cost / (float(inference_count) * 1.0f);
        sample::user_visible_stream_log("inference [", inference_count, "] times, total takes: ", infer_cost, "  milliseconds, average inference time: ", average_cost, ", frames per second: ", int(1000.0f / average_cost));
    }

    buffer_ptr_->copyOutputToHost();

    output_data.clear();
    float* output_data_ptr = (float*)buffer_ptr_->getHostBuffer(output_name_);
    for (int i = 0; i < output_tensor_size_; i++) {
        output_data.emplace_back(output_data_ptr[i]);
    }

    return true;
}
