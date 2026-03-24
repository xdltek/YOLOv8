/**
 * @file infer_sample_base.cpp
 * @brief Shared build/infer implementation for ONNX demo models.
 */
#include <filesystem>

#include "infer_sample_base.h"
#include "calibrator.h"
#include "parser_api.h"
#include "OnnxParser.h"
#include "opencv2/opencv.hpp"
#include "logger.h"

/**
 * @brief Build engine from ONNX, set precision/config options, and cache binding metadata.
 * @return true on success, false on failure.
 */
bool SampleModel::build() {
    // Legacy calibrator holder kept for compatibility with older sample flow.
    std::unique_ptr<infer1::IInt8Calibrator> calibrator;

    std::unique_ptr<IBuilder> builder {createInferBuilder(sample::gLogger.getLogger())};
    if (builder == nullptr)
    {
        sample::LOG_ERROR() << "Unable to create builder object." << std::endl;
        return false;
    }

    std::unique_ptr<INetworkDefinition> network { builder->createNetwork() };
    if (network == nullptr)
    {
        sample::LOG_ERROR() << "Unable to create network object." << std::endl;
        return false;
    }
    if(!mParams.model_config.empty())
    {
        dynamic_cast<infer1::INetworkDefinition *>(network.get())->setConfigFile(mParams.model_config.c_str());
    }

    std::unique_ptr<IBuilderConfig> config {builder->createBuilderConfig()};
    if (config == nullptr)
    {
        sample::LOG_ERROR() << "Unable to create config object." << std::endl;
        return false;
    }

    std::unique_ptr<onnxparser::IParser> parser { onnxparser::createParser(*network, sample::gLogger.getLogger()) };
    if (parser == nullptr)
    {
        sample::LOG_ERROR() << "Unable to parse ONNX model file: " << mParams.onnxFileName << std::endl;
        return false;
    }
    sample::user_visible_stream_log("onnx model: ", mParams.onnxFileName);

    if (onnx_parser(mParams.onnxFileName, builder.get(), network.get(), parser.get()) != 0) {
        return false;
    }

    if(!mParams.model_config.empty()) {
        network->setConfigFile(mParams.model_config.c_str());
    }

    // Build runtime engine with current config flags.
    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize((size_t) 1_GiB);
    if (mParams.bf16) {
        config->setFlag(BuilderFlag::kBF16);
    }

    mEngine = std::shared_ptr<infer1::IEngine>(builder->buildEngineWithConfig(*network, *config));
    if (!mEngine.get()) {
        return false;
    }

    auto has_flag = config->getFlag(infer1::BuilderFlag::kINT8);
    if (has_flag) {
        sample::user_visible_log("Since the model includes quantization layers, inference is performed with int8 precision.");
    }

    // Populate input/output binding names.
    getInputOutputNames();

    // Cache binding dimensions and input data type.
    const int inputIndex = mEngine->getBindingIndex(mInOut["input"].c_str());
    mInputDims = mEngine->getBindingDimensions(inputIndex);
    mInputDataType = mEngine->getBindingDataType(inputIndex);

    const int outputIndex = mEngine->getBindingIndex(mInOut["output"].c_str());
    mOutputDims = mEngine->getBindingDimensions(outputIndex);

    for(std::string& directory : mParams.dataDirs)
    {
        loadImageFileNames(directory);
    }

    config->destroy();
    network->destroy();
    builder->destroy();
    return true;
}

/**
 * @brief Execute preprocess -> warmup -> timed inference -> postprocess loop.
 * @return true on success, false on failure.
 */
bool SampleModel::infer() {
    if (!IsGlobalStopwatchRunning())
    {
        // SetGlobalStopwatchStatus(true);
    }

    ResetGlobalStopwatch();
    StartGlobalStopwatch();

    // Create buffer manager for host/device binding memory.
    samplesCommon::RppBufferManager buffers(mEngine);

    std::unique_ptr<infer1::IExecutionContext> context {mEngine->createExecutionContext()};
    if (!context)
    {
        return false;
    }
    if(!mParams.model_config.empty())
    {
        context->setModelConfigName(mParams.model_config.c_str());
    }
    float infer_cost = 0.0f;
    samplesCommon::PreciseCpuTimer infer_timer;

    long long current_loop = 0;
    rtStream_t rt_stream;
    rtStreamCreate(&rt_stream);

    if (mParams.image_count == 0) {
        sample::user_visible_stream_log("[ERROR] Cannot found any image file in directory 'image'");
        return false;
    }

    std::vector<float> cost_list;
    while(current_loop < mParams.image_count)
    {
        if (!PreProcess(buffers))
        {
            return false;
        }

        buffers.copyInputToDevice();

        // Warm up once before timed loops to reduce first-run overhead.
        bool ok = context->execute(1, buffers.getDeviceBindings().data());
        if (!ok)
        {
            throw std::runtime_error("execute failed.");
        }

        // Timed inference loop.
        for(int i = 0; i < mParams.loops; ++i)
        {
            infer_timer.reset();
            infer_timer.start();
            ok = context->execute(1, buffers.getDeviceBindings().data());
            if (!ok)
            {
                throw std::runtime_error("execute failed.");
            }

            infer_timer.stop();
            infer_cost = infer_timer.milliseconds();

            cost_list.emplace_back(infer_cost);

            StopGlobalStopwatch();
            ShowStopwatchInMicroseconds();
        }

        buffers.copyOutputToHost();
        PostProcess(buffers);
        current_loop++;
    }
    rtStreamDestroy(rt_stream);

    if (cost_list.empty()) {
        sample::LOG_ERROR() << "No inference operations performed. Please check parameters." << std::endl;
        return false;
    }

    if (cost_list.size() == 1) {
        sample::user_visible_stream_log("inference takes: ", infer_cost, "  milliseconds, frames per second: ", int(1000.0f / infer_cost));
    }
    else {
        const float total_cost = std::accumulate(cost_list.begin(), cost_list.end(), 0.0f);
        float average_cost = total_cost / static_cast<float>(cost_list.size());

        sample::user_visible_stream_log("inference [", cost_list.size(), "] times, total takes: ", total_cost, "  milliseconds, average inference time: ", average_cost, ", frames per second: ", int(1000.0f / average_cost));
    }

    return true;
}

/**
 * @brief Resolve engine bindings to logical "input"/"output" names.
 */
void SampleModel::getInputOutputNames() {
    int nbindings = mEngine->getNbBindings();
    ASSERT(nbindings == 2);
    for (int b = 0; b < nbindings; ++b)
    {
        infer1::Dims dims = mEngine->getBindingDimensions(b);
        if (mEngine->bindingIsInput(b))
        {
            mInOut["input"] = mEngine->getBindingName(b);
        }
        else
        {
            mInOut["output"] = mEngine->getBindingName(b);
        }
    }
}

/**
 * @brief Recursively load image file paths from a folder.
 * @param folder_name Folder to scan.
 */
void SampleModel::loadImageFileNames(const std::string &folder_name) {
    std::filesystem::directory_entry entry(folder_name);
    if (!(entry.exists() && entry.is_directory()))
    {
        return;
    }
    std::filesystem::directory_iterator iterator(folder_name);

    for (auto const& dir_entry : iterator) {

        std::filesystem::path directory_path = dir_entry.path();

        if (is_directory(directory_path))
        {
            loadImageFileNames(directory_path);
        }
        else
        {
            std::string file_ext = directory_path.extension();

            if (file_ext != ".jpg" && file_ext != ".jpeg" && file_ext != ".JPG" && file_ext != ".JPEG" && file_ext != ".png" && file_ext != ".bin") {
                continue;
            }

            std::string image_file_name = dir_entry.path();

//            sample::LOG_INFO() << "Image file: " << image_file_name << std::endl;
            image_file_names_.emplace_back(image_file_name);
        }
    }
}
