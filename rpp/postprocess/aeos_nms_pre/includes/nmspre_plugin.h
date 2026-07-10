/**
 * @file nmspre_plugin.h
 * @brief OpenRT plugin that splits YOLOv8 output into boxes and class scores before NMS.
 */
#pragma once

#include <string>

#include "InferRuntimeCommon.h"

/**
 * @brief OpenRT plugin used by YOLOv8 postprocessing before addNMS.
 *
 * The plugin accepts a YOLOv8 output matrix, extracts the first four box
 * channels into one output tensor, and writes class-score channels into a
 * second output tensor. Transposed model output layouts are supported.
 */
class NMSPREPlugin : public infer1::IPluginV2
{
public:
    /**
     * @brief Return the plugin type name registered with OpenRT.
     */
    const char *getPluginType() const ORTNOEXCEPT override;
    /**
     * @brief Return the plugin version string.
     */
    const char *getPluginVersion() const ORTNOEXCEPT override;
    /**
     * @brief Return the number of plugin outputs: boxes and scores.
     */
    int getNbOutputs() const ORTNOEXCEPT override;
    /**
     * @brief Resolve the shape of the requested output tensor.
     */
    infer1::Dims getOutputDimensions(int index, const infer1::Dims *inputs, int nbInputDims) ORTNOEXCEPT override;
    /**
     * @brief Report whether the plugin supports the requested tensor type and layout.
     */
    bool supportsFormat(infer1::DataType type, infer1::PluginFormat format) const ORTNOEXCEPT override;
    /**
     * @brief Capture input dimensions selected by OpenRT during engine build.
     */
    void configureWithFormat(const infer1::Dims *inputDims, int nbInputs, const infer1::Dims *outputDims, int nbOutputs,
                             infer1::DataType type, infer1::PluginFormat format, int maxBatchSize) ORTNOEXCEPT override;
    /**
     * @brief Initialize plugin resources.
     */
    int initialize() ORTNOEXCEPT override;
    /**
     * @brief Release plugin resources.
     */
    void terminate() ORTNOEXCEPT override;
    /**
     * @brief Return extra workspace bytes needed during enqueue.
     */
    size_t getWorkspaceSize(int maxBatchSize) const ORTNOEXCEPT override;
    /**
     * @brief Launch the PreNMS rearrange kernels.
     */
    int enqueue(int32_t batchSize, const void *const *inputs, void *const *outputs, void *workspace,
                rtStream_t stream) ORTNOEXCEPT override;
    /**
     * @brief Return serialized state size.
     */
    size_t getSerializationSize() const ORTNOEXCEPT override;
    /**
     * @brief Serialize plugin state.
     */
    void serialize(void *buffer) const ORTNOEXCEPT override;
    /**
     * @brief Destroy this plugin instance.
     */
    void destroy() ORTNOEXCEPT override;
    /**
     * @brief Clone this plugin for another OpenRT engine/context.
     */
    IPluginV2 *clone() const ORTNOEXCEPT override;
    /**
     * @brief Set the OpenRT plugin namespace.
     */
    void setPluginNamespace(const char *pluginNamespace) ORTNOEXCEPT override;
    /**
     * @brief Return the OpenRT plugin namespace.
     */
    const char *getPluginNamespace() const ORTNOEXCEPT override;

public:
    /**
     * @brief Set the input height used by the plugin.
     */
    void setHeight(int height) { height_ = height; }
    /**
     * @brief Set the input width used by the plugin.
     */
    void setWidth(int width) { width_ = width; }
    /**
     * @brief Return the configured input height.
     */
    int getHeight() const { return height_; }
    /**
     * @brief Return the configured input width.
     */
    int getWidth() const { return width_; }
    /**
     * @brief Configure thresholds and layout parameters used by the PreNMS kernels.
     */
    void setParams(float obj_threshold, float nms_threshold, int max_class, bool transpose, int type_size = 4);

private:
    static constexpr int kSramSize = 24 * 1024 * 1024;
    int height_ = 0;
    int width_ = 0;
    std::string namespace_;
    float obj_threshold_ = 0.25;
    float nms_threshold_ = 0.45;
    int max_class_ = 200;
    bool is_tanspose_ = false;
    int type_size_ = 4;
};

/**
 * @brief Factory object used by OpenRT to create NMSPREPlugin instances.
 */
class NMSPREPluginCreator : public infer1::IPluginCreator
{
public:
    /**
     * @brief Return the plugin name registered by this creator.
     */
    const char *getPluginName() const ORTNOEXCEPT override;
    /**
     * @brief Return the plugin version registered by this creator.
     */
    const char *getPluginVersion() const ORTNOEXCEPT override;
    /**
     * @brief Return plugin fields accepted by this creator.
     */
    const infer1::PluginFieldCollection *getFieldNames() ORTNOEXCEPT override;
    /**
     * @brief Create a new plugin instance.
     */
    infer1::IPluginV2 *createPlugin(const char *name, const infer1::PluginFieldCollection *fc) ORTNOEXCEPT override;
    /**
     * @brief Recreate a plugin instance from serialized state.
     */
    infer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) ORTNOEXCEPT override;
    /**
     * @brief Set the namespace for plugins produced by this creator.
     */
    void setPluginNamespace(const char *pluginNamespace) ORTNOEXCEPT override;
    /**
     * @brief Return the namespace for plugins produced by this creator.
     */
    const char *getPluginNamespace() const ORTNOEXCEPT override;

private:
    std::string namespace_;
    std::string plugin_name_ = "NMSPREPlugin";
    infer1::PluginFieldCollection field_collection_{ 0, nullptr };
};
