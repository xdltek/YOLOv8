/**
 * @file rpp_yolo_preprocessor.cpp
 * @brief Reusable RPP YOLO preprocessing implementation.
 */
#include "rpp_yolo_preprocessor.h"

#include "rpp_preprocess_kernels.cuh"
#include "perf_trace_session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

namespace
{
using ProfileClock = std::chrono::high_resolution_clock;

/**
 * @brief Check whether a preprocessing format is part of the YUV family.
 * @param format Source input layout.
 */
bool is_yuv_format(PreprocessInputFormat format)
{
    return format == PreprocessInputFormat::YUV_I420 ||
           format == PreprocessInputFormat::NV12 ||
           format == PreprocessInputFormat::NV21;
}

/**
 * @brief Check whether the RPP YOLO preprocessing path has an implementation for the format.
 * @param format Source input layout.
 */
bool is_supported_format(PreprocessInputFormat format)
{
    return format == PreprocessInputFormat::YUV_I420 ||
           format == PreprocessInputFormat::RGB_HWC ||
           format == PreprocessInputFormat::BGR_HWC ||
           format == PreprocessInputFormat::RGB_CHW;
}

/**
 * @brief Print a runtime API error and convert the status into a boolean.
 * @param status Return value from an RPP runtime call.
 * @param what Human-readable operation name for diagnostics.
 */
bool check_rt(rtError_t status, const char* what)
{
    if (status == rtError_t::rtSuccess) {
        return true;
    }
    std::cerr << what << " failed, rtError=" << static_cast<int>(status) << std::endl;
    return false;
}

/**
 * @brief Measure elapsed milliseconds from a saved timestamp to now.
 * @param start Timestamp captured before a measured stage.
 */
double elapsed_ms(ProfileClock::time_point start)
{
    return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}

/**
 * @brief Clear an optional preprocessing profile before filling current-run timings.
 * @param profile Optional profile object owned by the caller.
 */
void reset_profile(RppPreprocessProfile* profile)
{
    if (profile != nullptr) {
        *profile = RppPreprocessProfile{};
    }
}

}

/**
 * @brief Release persistent host, DDR, and SRAM preprocessing allocations.
 */
RppYoloPreprocessor::~RppYoloPreprocessor()
{
    // Release persistent DDR and pinned host staging buffers owned by this preprocessor.
    if (host_input_device_ != nullptr) {
        rtFree(host_input_device_);
        host_input_device_ = nullptr;
    }
    if (host_input_pinned_ != nullptr) {
        rtFreeHost(host_input_pinned_);
        host_input_pinned_ = nullptr;
    }
    releaseSramBuffers();
}

/**
 * @brief Release SRAM workspace owned by preprocessing.
 */
void RppYoloPreprocessor::releaseSramBuffers()
{
    // SRAM is temporary workspace; release it after preprocessing so postprocess can allocate its own workspace.
    if (input_sram_ != nullptr) {
        rtFreeSram(input_sram_);
        input_sram_ = nullptr;
        input_sram_capacity_ = 0;
    }
    if (rgb_chw_sram_ != nullptr) {
        rtFreeSram(rgb_chw_sram_);
        rgb_chw_sram_ = nullptr;
        rgb_chw_sram_capacity_ = 0;
    }
    if (model_input_sram_ != nullptr) {
        rtFreeSram(model_input_sram_);
        model_input_sram_ = nullptr;
        model_input_sram_capacity_ = 0;
    }
}

/**
 * @brief Store preprocessing limits and model input dimensions.
 */
bool RppYoloPreprocessor::init(int max_input_width, int max_input_height, int model_width, int model_height)
{
    // Store shape limits up front so each run can validate input cheaply.
    if (max_input_width <= 0 || max_input_height <= 0 || model_width <= 0 || model_height <= 0) {
        std::cerr << "Invalid RPP preprocessor dimensions." << std::endl;
        return false;
    }

    max_input_width_ = max_input_width;
    max_input_height_ = max_input_height;
    model_width_ = model_width;
    model_height_ = model_height;
    return true;
}

/**
 * @brief Execute one source frame through H2D staging, RPP resize/normalize, and model-input copy.
 */
bool RppYoloPreprocessor::run(const PreprocessInput& input,
                              void* model_input_device,
                              LetterboxInfo& letterbox,
                              rtStream_t stream,
                              RppPreprocessProfile* profile)
{
    PERF_SCOPE_CATE("rpp_yolo_preprocess_run", "preprocess");
    reset_profile(profile);
    // Validate both the input metadata and the model input destination before allocating workspace.
    if (!validateInput(input) || model_input_device == nullptr) {
        return false;
    }

    // Reuse an external stream when the pipeline wants to compose stages; otherwise own the stream locally.
    rtStream_t run_stream = stream;
    bool owns_stream = false;
    if (run_stream == nullptr) {
        if (!check_rt(rtStreamCreate(&run_stream), "rtStreamCreate")) {
            return false;
        }
        owns_stream = true;
    }

    // Stage source data in device DDR first; device inputs can pass through without another H2D copy.
    void* device_input = nullptr;
    auto stage_start = ProfileClock::now();
    if (!copyHostInputToDevice(input, run_stream, &device_input)) {
        if (owns_stream) {
            rtStreamDestroy(run_stream);
        }
        return false;
    }
    if (profile != nullptr) {
        profile->host_to_device_ms = input.memory_type == PreprocessMemoryType::Host ? elapsed_ms(stage_start) : 0.0;
        profile->host_to_device_bytes = input.memory_type == PreprocessMemoryType::Host ?
                                        rpp_preprocess_input_bytes(input.format, input.width, input.height) :
                                        0U;
    }
    auto preprocess_start = ProfileClock::now();

    // Move input DDR into SRAM because the RPP kernels operate on temporary SRAM workspace.
    size_t input_bytes = rpp_preprocess_input_bytes(input.format, input.width, input.height);
    if (!ensureSramBuffer(&input_sram_, &input_sram_capacity_, input_bytes)) {
        if (owns_stream) {
            rtStreamDestroy(run_stream);
        }
        return false;
    }
    if (!check_rt(rtMemcpy(input_sram_, device_input, input_bytes, rtMemcpyDeviceToSram), "rtMemcpy input DDR to SRAM")) {
        if (owns_stream) {
            rtStreamDestroy(run_stream);
        }
        return false;
    }

    // Save letterbox metadata so postprocessing can map boxes back to the original source size.
    letterbox = make_letterbox_info(input.width, input.height, model_width_, model_height_);

    size_t model_input_bytes = static_cast<size_t>(model_width_) *
                               static_cast<size_t>(model_height_) *
                               3U *
                               sizeof(float);
    if (!ensureSramBuffer(&model_input_sram_, &model_input_sram_capacity_, model_input_bytes)) {
        if (owns_stream) {
            rtStreamDestroy(run_stream);
        }
        return false;
    }
    if (profile != nullptr) {
        profile->model_input_clear_ms = 0.0;
    }

    // Launch the format-specific resize/normalize kernel and write NCHW float data in SRAM.
    stage_start = ProfileClock::now();
    if (input.format == PreprocessInputFormat::YUV_I420) {
        launch_letterbox_resize_norm_i420_to_nchw_f32(run_stream,
                                                      input_sram_,
                                                      input.width,
                                                      input.height,
                                                      static_cast<float*>(model_input_sram_),
                                                      model_width_,
                                                      model_height_,
                                                      letterbox.resized_width,
                                                      letterbox.resized_height,
                                                      letterbox.scale,
                                                      letterbox.pad_x,
                                                      letterbox.pad_y);
    }
    else if (input.format == PreprocessInputFormat::RGB_CHW) {
        launch_letterbox_resize_norm_chw_u8_to_nchw_f32(run_stream,
                                                        input_sram_,
                                                        input.width,
                                                        input.height,
                                                        static_cast<float*>(model_input_sram_),
                                                        model_width_,
                                                        model_height_,
                                                        letterbox.resized_width,
                                                        letterbox.resized_height,
                                                        letterbox.scale,
                                                        letterbox.pad_x,
                                                        letterbox.pad_y);
    }
    else {
        bool input_is_bgr = input.format == PreprocessInputFormat::BGR_HWC;
        launch_letterbox_resize_norm_hwc_u8_to_nchw_f32(run_stream,
                                                        input_sram_,
                                                        input.width,
                                                        input.height,
                                                        input_is_bgr,
                                                        static_cast<float*>(model_input_sram_),
                                                        model_width_,
                                                        model_height_,
                                                        letterbox.resized_width,
                                                        letterbox.resized_height,
                                                        letterbox.scale,
                                                        letterbox.pad_x,
                                                        letterbox.pad_y);
    }

    if (!check_rt(rtGetLastError(), "RPP preprocessing kernel launch")) {
        if (owns_stream) {
            rtStreamDestroy(run_stream);
        }
        return false;
    }

    if (!check_rt(rtStreamSynchronize(run_stream), "rtStreamSynchronize preprocess kernels")) {
        if (owns_stream) {
            rtStreamDestroy(run_stream);
        }
        return false;
    }
    if (profile != nullptr) {
        profile->resize_normalize_ms = elapsed_ms(stage_start);
    }
    // Copy normalized model input from SRAM back to the runtime binding in device DDR.
    if (!check_rt(rtMemcpy(model_input_device, model_input_sram_, model_input_bytes, rtMemcpySramToDevice),
                  "rtMemcpy model input SRAM to DDR")) {
        if (owns_stream) {
            rtStreamDestroy(run_stream);
        }
        return false;
    }
    if (profile != nullptr) {
        profile->total_ms = elapsed_ms(preprocess_start);
    }

    bool ok = true;
    if (owns_stream) {
        check_rt(rtStreamDestroy(run_stream), "rtStreamDestroy");
    }
    return ok;
}

/**
 * @brief Grow or reuse a device DDR buffer for preprocessing input staging.
 */
bool RppYoloPreprocessor::ensureDeviceBuffer(void** buffer, size_t* capacity, size_t required_bytes)
{
    // Reuse the current allocation when it is already large enough for this frame.
    if (buffer == nullptr || capacity == nullptr || required_bytes == 0) {
        return false;
    }
    if (*buffer != nullptr && *capacity >= required_bytes) {
        return true;
    }
    if (*buffer != nullptr) {
        if (!check_rt(rtFree(*buffer), "rtFree")) {
            return false;
        }
        *buffer = nullptr;
        *capacity = 0;
    }
    if (!check_rt(rtMalloc(buffer, required_bytes), "rtMalloc")) {
        return false;
    }
    *capacity = required_bytes;
    return true;
}

/**
 * @brief Grow or reuse a pinned host buffer for asynchronous H2D transfer.
 */
bool RppYoloPreprocessor::ensureHostPinnedBuffer(void** buffer, size_t* capacity, size_t required_bytes)
{
    // Pinned host memory is used as a stable asynchronous H2D staging area.
    if (buffer == nullptr || capacity == nullptr || required_bytes == 0) {
        return false;
    }
    if (*buffer != nullptr && *capacity >= required_bytes) {
        return true;
    }
    if (*buffer != nullptr) {
        if (!check_rt(rtFreeHost(*buffer), "rtFreeHost")) {
            return false;
        }
        *buffer = nullptr;
        *capacity = 0;
    }
    if (!check_rt(rtHostAlloc(buffer, required_bytes, 0), "rtHostAlloc preprocess input")) {
        return false;
    }
    *capacity = required_bytes;
    return true;
}

/**
 * @brief Grow or reuse an SRAM workspace buffer for RPP kernels.
 */
bool RppYoloPreprocessor::ensureSramBuffer(void** buffer, size_t* capacity, size_t required_bytes)
{
    // SRAM buffers are resized lazily because source image size can change between runs.
    if (buffer == nullptr || capacity == nullptr || required_bytes == 0) {
        return false;
    }
    if (*buffer != nullptr && *capacity >= required_bytes) {
        return true;
    }
    if (*buffer != nullptr) {
        if (!check_rt(rtFreeSram(*buffer), "rtFreeSram")) {
            return false;
        }
        *buffer = nullptr;
        *capacity = 0;
    }
    if (!check_rt(rtMallocSram(buffer, required_bytes), "rtMallocSram")) {
        return false;
    }
    *capacity = required_bytes;
    return true;
}

/**
 * @brief Copy host input into device DDR or forward an existing device input pointer.
 */
bool RppYoloPreprocessor::copyHostInputToDevice(const PreprocessInput& input,
                                                rtStream_t stream,
                                                void** device_input)
{
    // Device-backed inputs are already in the address space expected by downstream stages.
    if (device_input == nullptr) {
        return false;
    }
    if (input.memory_type == PreprocessMemoryType::Device) {
        *device_input = const_cast<void*>(input.data);
        return true;
    }

    // Copy exactly the bytes required by the declared source format.
    size_t required = rpp_preprocess_input_bytes(input.format, input.width, input.height);
    if (input.bytes < required) {
        std::cerr << "Input bytes are smaller than required preprocessing bytes." << std::endl;
        return false;
    }
    if (!ensureDeviceBuffer(&host_input_device_, &host_input_capacity_, required)) {
        return false;
    }
    if (!ensureHostPinnedBuffer(&host_input_pinned_, &host_input_pinned_capacity_, required)) {
        return false;
    }
    std::memcpy(host_input_pinned_, input.data, required);
    if (!check_rt(rtMemcpyAsync(host_input_device_,
                                host_input_pinned_,
                                required,
                                rtMemcpyHostToDevice,
                                stream),
                  "rtMemcpyAsync preprocess H2D") ||
        !check_rt(rtStreamSynchronize(stream), "rtStreamSynchronize preprocess H2D")) {
        return false;
    }
    *device_input = host_input_device_;
    return true;
}

/**
 * @brief Validate source metadata before preprocessing allocates runtime resources.
 */
bool RppYoloPreprocessor::validateInput(const PreprocessInput& input) const
{
    // Reject malformed shape and buffer metadata before touching runtime memory.
    if (input.data == nullptr || input.width <= 0 || input.height <= 0) {
        std::cerr << "Invalid preprocessing input." << std::endl;
        return false;
    }
    if (input.width > max_input_width_ || input.height > max_input_height_) {
        std::cerr << "Preprocessing input exceeds initialized maximum dimensions." << std::endl;
        return false;
    }
    if (is_yuv_format(input.format) && input.format != PreprocessInputFormat::YUV_I420) {
        std::cerr << "Unsupported YUV format. Only I420 is implemented." << std::endl;
        return false;
    }
    if (input.format == PreprocessInputFormat::YUV_I420 &&
        ((input.width % 2) != 0 || (input.height % 2) != 0)) {
        std::cerr << "I420 input width and height must be even." << std::endl;
        return false;
    }
    if (!is_supported_format(input.format)) {
        std::cerr << "Unsupported preprocessing input format." << std::endl;
        return false;
    }
    size_t required = rpp_preprocess_input_bytes(input.format, input.width, input.height);
    if (required == 0 || input.bytes < required) {
        std::cerr << "Invalid preprocessing input byte size." << std::endl;
        return false;
    }
    return true;
}

/**
 * @brief Return required raw input bytes for the selected preprocessing format.
 */
size_t rpp_preprocess_input_bytes(PreprocessInputFormat format, int width, int height)
{
    // The byte count must match the raw host buffer layout accepted by each demo.
    if (width <= 0 || height <= 0) {
        return 0;
    }
    size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    switch (format) {
        case PreprocessInputFormat::YUV_I420:
        case PreprocessInputFormat::NV12:
        case PreprocessInputFormat::NV21:
            return pixels * 3U / 2U;
        case PreprocessInputFormat::RGB_HWC:
        case PreprocessInputFormat::BGR_HWC:
        case PreprocessInputFormat::RGB_CHW:
            return pixels * 3U;
        default:
            return 0;
    }
}

/**
 * @brief Compute YOLO letterbox geometry for a source image and model input.
 */
LetterboxInfo make_letterbox_info(int source_width, int source_height, int model_width, int model_height)
{
    // Match YOLO letterbox behavior: preserve aspect ratio and center the resized image.
    LetterboxInfo info;
    info.source_width = source_width;
    info.source_height = source_height;
    info.model_width = model_width;
    info.model_height = model_height;

    float scale_w = static_cast<float>(model_width) / static_cast<float>(source_width);
    float scale_h = static_cast<float>(model_height) / static_cast<float>(source_height);
    info.scale = std::min(scale_w, scale_h);
    info.resized_width = std::max(1, static_cast<int>(std::round(static_cast<float>(source_width) * info.scale)));
    info.resized_height = std::max(1, static_cast<int>(std::round(static_cast<float>(source_height) * info.scale)));
    float half_pad_x = (static_cast<float>(model_width) - static_cast<float>(info.resized_width)) * 0.5f;
    float half_pad_y = (static_cast<float>(model_height) - static_cast<float>(info.resized_height)) * 0.5f;
    info.pad_x = static_cast<float>(static_cast<int>(std::round(half_pad_x - 0.1f)));
    info.pad_y = static_cast<float>(static_cast<int>(std::round(half_pad_y - 0.1f)));
    return info;
}
