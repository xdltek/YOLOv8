/**
 * @file rpp_yolo_preprocessor.h
 * @brief Reusable RPP YOLO preprocessing interface.
 */
#ifndef XDLTEK_SAMPLES_RPP_YOLO_PREPROCESSOR_H
#define XDLTEK_SAMPLES_RPP_YOLO_PREPROCESSOR_H

#include "detection_types.h"

#include <rpp_runtime.h>

#include <cstddef>

/**
 * @brief Supported source layouts accepted by the RPP YOLO preprocessor.
 */
enum class PreprocessInputFormat
{
    YUV_I420,
    RGB_HWC,
    BGR_HWC,
    RGB_CHW,
    NV12,
    NV21,
};

/**
 * @brief Location of the source buffer passed to preprocessing.
 */
enum class PreprocessMemoryType
{
    Host,
    Device,
};

/**
 * @brief Source frame descriptor consumed by one preprocessing run.
 */
struct PreprocessInput
{
    const void* data = nullptr;
    int width = 0;
    int height = 0;
    PreprocessInputFormat format = PreprocessInputFormat::BGR_HWC;
    PreprocessMemoryType memory_type = PreprocessMemoryType::Host;
    size_t bytes = 0;
};

/**
 * @brief Timing and transfer counters reported by the preprocessing module.
 */
struct RppPreprocessProfile
{
    double host_to_device_ms = 0.0;
    size_t host_to_device_bytes = 0;
    double model_input_clear_ms = 0.0;
    double yuv_to_rgb_ms = 0.0;
    double resize_normalize_ms = 0.0;
    double total_ms = 0.0;
};

/**
 * @brief RPP-backed YOLO letterbox, colorspace conversion, resize, and normalization module.
 */
class RppYoloPreprocessor
{
public:
    /**
     * @brief Construct an empty preprocessor; call init() before run().
     */
    RppYoloPreprocessor() = default;
    /**
     * @brief Release all persistent preprocessing allocations.
     */
    ~RppYoloPreprocessor();

    /**
     * @brief Disable copying because the object owns runtime memory buffers.
     */
    RppYoloPreprocessor(const RppYoloPreprocessor&) = delete;
    /**
     * @brief Disable copy assignment because the object owns runtime memory buffers.
     */
    RppYoloPreprocessor& operator=(const RppYoloPreprocessor&) = delete;

    /**
     * @brief Configure accepted source dimensions and model input shape.
     * @param max_input_width Maximum source width this instance may process.
     * @param max_input_height Maximum source height this instance may process.
     * @param model_width YOLO model input width.
     * @param model_height YOLO model input height.
     */
    bool init(int max_input_width, int max_input_height, int model_width, int model_height);

    /**
     * @brief Run RPP YOLO preprocessing into the model input device buffer.
     * @param input Host or device input descriptor.
     * @param model_input_device Destination model input buffer in device memory.
     * @param letterbox Output metadata needed to restore boxes after inference.
     * @param stream Optional runtime stream.
     * @param profile Optional timing profile populated for demo logs.
     */
    bool run(const PreprocessInput& input,
             void* model_input_device,
             LetterboxInfo& letterbox,
             rtStream_t stream = nullptr,
             RppPreprocessProfile* profile = nullptr);

    /**
     * @brief Release SRAM buffers owned by preprocessing.
     */
    void releaseSramBuffers();

private:
    /**
     * @brief Ensure a reusable device DDR buffer has at least the requested capacity.
     * @param buffer Buffer pointer updated when allocation changes.
     * @param capacity Current capacity updated on allocation.
     * @param required_bytes Required byte count.
     */
    bool ensureDeviceBuffer(void** buffer, size_t* capacity, size_t required_bytes);
    /**
     * @brief Ensure a reusable pinned host buffer has at least the requested capacity.
     * @param buffer Buffer pointer updated when allocation changes.
     * @param capacity Current capacity updated on allocation.
     * @param required_bytes Required byte count.
     */
    bool ensureHostPinnedBuffer(void** buffer, size_t* capacity, size_t required_bytes);
    /**
     * @brief Ensure a reusable SRAM workspace buffer has at least the requested capacity.
     * @param buffer Buffer pointer updated when allocation changes.
     * @param capacity Current capacity updated on allocation.
     * @param required_bytes Required byte count.
     */
    bool ensureSramBuffer(void** buffer, size_t* capacity, size_t required_bytes);
    /**
     * @brief Copy host input into device DDR, or pass through an existing device pointer.
     * @param input Source input descriptor.
     * @param stream Runtime stream used by asynchronous H2D copy.
     * @param device_input Output device pointer consumed by preprocessing.
     */
    bool copyHostInputToDevice(const PreprocessInput& input, rtStream_t stream, void** device_input);
    /**
     * @brief Validate source shape, format, and byte count before preprocessing.
     * @param input Source input descriptor.
     */
    bool validateInput(const PreprocessInput& input) const;

    int max_input_width_ = 0;
    int max_input_height_ = 0;
    int model_width_ = 0;
    int model_height_ = 0;

    void* host_input_device_ = nullptr;
    size_t host_input_capacity_ = 0;
    void* host_input_pinned_ = nullptr;
    size_t host_input_pinned_capacity_ = 0;
    void* input_sram_ = nullptr;
    size_t input_sram_capacity_ = 0;
    void* rgb_chw_sram_ = nullptr;
    size_t rgb_chw_sram_capacity_ = 0;
    void* model_input_sram_ = nullptr;
    size_t model_input_sram_capacity_ = 0;
};

/**
 * @brief Compute required source bytes for a supported preprocessing input format.
 * @param format Source input layout.
 * @param width Source width.
 * @param height Source height.
 */
size_t rpp_preprocess_input_bytes(PreprocessInputFormat format, int width, int height);

/**
 * @brief Compute YOLO letterbox scale and padding used by preprocessing and postprocessing.
 * @param source_width Source image or frame width.
 * @param source_height Source image or frame height.
 * @param model_width Model input width.
 * @param model_height Model input height.
 */
LetterboxInfo make_letterbox_info(int source_width, int source_height, int model_width, int model_height);

#endif // XDLTEK_SAMPLES_RPP_YOLO_PREPROCESSOR_H
