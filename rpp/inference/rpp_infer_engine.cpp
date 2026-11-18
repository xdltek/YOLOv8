/**
 * @file rpp_infer_engine.cpp
 * @brief Reusable OpenRT inference engine wrapper implementation.
 */
#include "rpp_infer_engine.h"

#include "logger.h"
#include "parser_api.h"
#include "sampleCommon.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

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

std::string modelFingerprint(const std::string& model_path)
{
    std::ifstream input(model_path, std::ios::binary);
    if (!input) {
        return {};
    }

    std::uint64_t hash = 14695981039346656037ULL;
    char buffer[64 * 1024];
    while (input.read(buffer, sizeof(buffer)) || input.gcount() > 0) {
        for (std::streamsize i = 0; i < input.gcount(); ++i) {
            hash ^= static_cast<unsigned char>(buffer[i]);
            hash *= 1099511628211ULL;
        }
    }

    std::ostringstream value;
    value << std::hex << std::setfill('0') << std::setw(16) << hash;
    return value.str();
}

std::filesystem::path engineCachePath(const std::string& model_path, const std::string& fingerprint)
{
    std::filesystem::path model(model_path);
    std::ostringstream filename;
    filename << model.filename().string() << ".openrt" << OPENRT_VERSION
             << ".bf16." << fingerprint << ".engine";
    return model.parent_path() / ".engine_cache" / filename.str();
}

std::string dataFingerprint(const void* data, size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    std::ostringstream value;
    value << std::hex << std::setfill('0') << std::setw(16) << hash;
    return value.str();
}

bool loadCachedEngine(const std::filesystem::path& cache_path,
                      std::unique_ptr<infer1::IRuntime>& runtime,
                      std::shared_ptr<infer1::IEngine>& engine)
{
    std::ifstream input(cache_path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0) {
        return false;
    }

    const std::streamsize size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    if (!input.read(data.data(), size)) {
        return false;
    }

    std::ifstream checksum_input(cache_path.string() + ".fnv1a");
    std::string expected_fingerprint;
    checksum_input >> expected_fingerprint;
    if (expected_fingerprint.empty()
        || expected_fingerprint != dataFingerprint(data.data(), data.size())) {
        std::cout << "[ENGINE  ] Engine cache integrity check failed; rebuilding from ONNX: "
                  << cache_path << std::endl;
        return false;
    }

    runtime.reset(infer1::createInferRuntime(sample::gLogger.getLogger()));
    if (!runtime) {
        return false;
    }

    sample::gLogger.clearRuntimeErrorCount();
    try {
        engine.reset(runtime->deserializeCudaEngine(data.data(), data.size()));
    }
    catch (const std::exception& error) {
        engine.reset();
        runtime.reset();
        std::cout << "[ENGINE  ] Cached engine is invalid (" << error.what()
                  << "); rebuilding from ONNX: " << cache_path << std::endl;
        return false;
    }
    if (!engine || sample::gLogger.getRuntimeErrorCount() != 0) {
        engine.reset();
        runtime.reset();
        std::cout << "[ENGINE  ] Cached engine could not be restored safely; rebuilding from ONNX: "
                  << cache_path << std::endl;
        return false;
    }
    return true;
}

bool saveEngineCache(const std::filesystem::path& cache_path, const infer1::IEngine& engine)
{
    std::unique_ptr<infer1::IHostMemory> serialized(engine.serialize());
    if (!serialized || serialized->data() == nullptr || serialized->size() == 0) {
        return false;
    }

    std::error_code error;
    std::filesystem::create_directories(cache_path.parent_path(), error);
    if (error) {
        return false;
    }

    const std::filesystem::path temporary_path = cache_path.string() + ".tmp";
    const std::filesystem::path checksum_path = cache_path.string() + ".fnv1a";
    const std::filesystem::path checksum_temporary_path = checksum_path.string() + ".tmp";
    std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
    if (!output.write(static_cast<const char*>(serialized->data()),
                      static_cast<std::streamsize>(serialized->size()))) {
        return false;
    }
    output.close();
    std::ofstream checksum_output(checksum_temporary_path, std::ios::trunc);
    checksum_output << dataFingerprint(serialized->data(), serialized->size()) << '\n';
    if (!checksum_output) {
        return false;
    }
    checksum_output.close();
    std::filesystem::rename(temporary_path, cache_path, error);
    if (error) {
        return false;
    }
    std::filesystem::rename(checksum_temporary_path, checksum_path, error);
    return !error;
}
}

/**
 * @brief Build the OpenRT engine from ONNX and prepare device binding buffers.
 */
bool RppInferEngine::init()
{
    const std::string fingerprint = modelFingerprint(onnx_model_path_);
    if (fingerprint.empty()) {
        std::cerr << "Unable to fingerprint ONNX model: " << onnx_model_path_ << std::endl;
        return false;
    }
    const std::filesystem::path cache_path = engineCachePath(onnx_model_path_, fingerprint);

    if (!rebuild_engine_ && loadCachedEngine(cache_path, runtime_ptr_, engine_ptr_)) {
        std::cout << "[ENGINE  ] Reusing cached OpenRT engine: " << cache_path << std::endl;
    }
    else {
        if (rebuild_engine_) {
            std::cout << "[ENGINE  ] Forced rebuild from complete ONNX; engine cache is disabled: "
                      << onnx_model_path_ << std::endl;
        }
        else {
            std::cout << "[ENGINE  ] No reusable engine cache found; building from ONNX: "
                      << onnx_model_path_ << std::endl;
        }
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

    // Forced rebuild preserves the complete original ONNX graph and the same
    // BF16 build policy; only persistent engine-cache reuse is bypassed.
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

        if (rebuild_engine_) {
            std::cout << "[ENGINE  ] Fresh BF16 engine is memory-only; cache write skipped." << std::endl;
        }
        else if (saveEngineCache(cache_path, *engine_ptr_)) {
            std::cout << "[ENGINE  ] Built and cached OpenRT engine: " << cache_path << std::endl;
        }
        else {
            std::cout << "[ENGINE  ] Built OpenRT engine; cache could not be written: "
                      << cache_path << std::endl;
        }
    }

    // Cache binding names, shapes, data types, and element counts for pipeline stages.
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
