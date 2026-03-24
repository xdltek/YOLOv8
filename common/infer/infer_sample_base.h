/**
 * @file infer_sample_base.h
 * @brief Base class for ONNX model build/inference flow in this demo.
 */
#pragma once

#include "Infer.h"
#include "argsParser.h"
#include "rpp_buffer_manager.h"
#include <memory>

using namespace infer1;

/**
 * @brief Common base for model build, preprocess/infer/postprocess, and buffer handling.
 */
class SampleModel
{
public:
    explicit SampleModel(const samplesCommon::ModelSampleParams& params)
            : mParams(params)
            , mEngine(nullptr), mInputDataType(DataType::kFLOAT)
    {
    }

    /**
     * @brief Build runtime engine from ONNX model and initialize I/O bindings.
     */
    bool build();

    /**
     * @brief Run inference loop for configured image inputs.
     */
    bool infer();

    bool teardown() { return true; }

public:
    samplesCommon::ModelSampleParams mParams; //!< Runtime parameters for this model instance.

    std::map<std::string, std::string> mInOut; //!< Tensor name mapping for "input" and "output".

    infer1::DataType mInputDataType; //!< Input tensor data type.

    infer1::Dims mInputDims; //!< Input tensor dimensions.

    infer1::Dims mOutputDims; //!< Output tensor dimensions.

    int mNumber{5}; //!< Legacy field from classification samples (not used in current YOLO demo).
    std::vector<std::string> mReferenceVector; //!< Legacy reference cache (not used in current YOLO demo).
    std::vector<std::string> image_file_names_; //!< Cached image file paths from configured data directories.
private:
    /**
     * @brief Populate mInOut map from engine bindings.
     */
    void getInputOutputNames();

    std::shared_ptr<infer1::IEngine> mEngine; //!< Runtime engine used for execution.


    bool warmup_ {false}; //!< Legacy warmup flag (currently unused).

    /**
     * @brief Postprocess model outputs and write visualization/results.
     * @param buffers Buffer manager containing output tensors on host.
     */
    virtual bool PostProcess(const samplesCommon::RppBufferManager& buffers) = 0;

    /**
     * @brief Load and preprocess one input image into the input binding buffer.
     * @param buffers Buffer manager containing input tensor memory on host.
     */
    virtual bool PreProcess(const samplesCommon::RppBufferManager& buffers) = 0;

    /**
     * @brief Recursively collect supported image files under a directory.
     * @param folder_name Root folder to scan.
     */
    void loadImageFileNames(const std::string& folder_name);
};