/**
 * @file rpp_preprocess_kernels.cuh
 * @brief RPP preprocessing kernel launcher declarations for YOLO inputs.
 */
#ifndef XDLTEK_SAMPLES_RPP_PREPROCESS_KERNELS_CUH
#define XDLTEK_SAMPLES_RPP_PREPROCESS_KERNELS_CUH

#include <rpp_runtime.h>

/**
 * @brief Letterbox-resize I420 input and normalize it into NCHW float model layout.
 */
void launch_letterbox_resize_norm_i420_to_nchw_f32(rtStream_t stream,
                                                   const void* yuv,
                                                   int input_width,
                                                   int input_height,
                                                   float* output,
                                                   int output_width,
                                                   int output_height,
                                                   int resized_width,
                                                   int resized_height,
                                                   float scale,
                                                   float pad_x,
                                                   float pad_y);

/**
 * @brief Letterbox-resize interleaved RGB/BGR HWC input and normalize it into NCHW float model layout.
 */
void launch_letterbox_resize_norm_hwc_u8_to_nchw_f32(rtStream_t stream,
                                                     const void* input,
                                                     int input_width,
                                                     int input_height,
                                                     bool input_is_bgr,
                                                     float* output,
                                                     int output_width,
                                                     int output_height,
                                                     int resized_width,
                                                     int resized_height,
                                                     float scale,
                                                     float pad_x,
                                                     float pad_y);

/**
 * @brief Letterbox-resize planar RGB CHW input and normalize it into NCHW float model layout.
 */
void launch_letterbox_resize_norm_chw_u8_to_nchw_f32(rtStream_t stream,
                                                     const void* input,
                                                     int input_width,
                                                     int input_height,
                                                     float* output,
                                                     int output_width,
                                                     int output_height,
                                                     int resized_width,
                                                     int resized_height,
                                                     float scale,
                                                     float pad_x,
                                                     float pad_y);

#endif // XDLTEK_SAMPLES_RPP_PREPROCESS_KERNELS_CUH
