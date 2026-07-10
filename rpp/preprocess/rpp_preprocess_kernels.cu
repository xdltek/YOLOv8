/**
 * @file rpp_preprocess_kernels.cu
 * @brief RPP-like CUDA kernels for YOLO letterbox resize and normalization.
 */
#include "rpp_preprocess_kernels.cuh"

#include <__clang_cuda_builtin_vars.h>
#include <drvapi_error_string.h>
#include <rpp_com.h>
#include <rpp_drv_api.h>
#include <rpp_math.h>
#include <rpp_runtime.h>

/**
 * @brief Resize, letterbox, convert I420 to RGB, and normalize into NCHW float output.
 */
__global__ void letterbox_resize_norm_i420_kernel(const unsigned char* y_base,
                                                  const unsigned char* u_base,
                                                  const unsigned char* v_base,
                                                  int input_width,
                                                  int input_height,
                                                  float* output,
                                                  int output_width,
                                                  int output_height,
                                                  int resized_width,
                                                  int resized_height,
                                                  int pad_left,
                                                  int pad_top,
                                                  int content_only)
{
    // Each launch lane writes one output pixel for one NCHW channel.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    // Content-only launches already point at the top-left model tensor content offset.
    // This scalar branch removes padding predicates from the common optimized path.
    float scale_x = static_cast<float>(input_width) / static_cast<float>(resized_width);
    float scale_y = static_cast<float>(input_height) / static_cast<float>(resized_height);
    int sample_x = x;
    int sample_y = y;
    short inside = 1;
    if (content_only == 0) {
        int rel_x = x - pad_left;
        int rel_y = y - pad_top;
        inside = static_cast<short>((rel_x >= 0) & (rel_x < resized_width) &
                                    (rel_y >= 0) & (rel_y < resized_height));
        // Clamp sampling coordinates so padding lanes never read outside the source frame.
        sample_x = rpp_min(rpp_max(rel_x, 0), resized_width - 1);
        sample_y = rpp_min(rpp_max(rel_y, 0), resized_height - 1);
    }

    int sx = static_cast<int>((static_cast<float>(sample_x) + 0.5f) * scale_x);
    int sy = static_cast<int>((static_cast<float>(sample_y) + 0.5f) * scale_y);
    sx = rpp_min(rpp_max(sx, 0), input_width - 1);
    sy = rpp_min(rpp_max(sy, 0), input_height - 1);

    int uv_width = input_width / 2;
    int y_offset = sy * input_width + sx;
    int uv_offset = (sy / 2) * uv_width + (sx / 2);

    unsigned char Y = y_base[y_offset];
    unsigned char U = u_base[uv_offset];
    unsigned char V = v_base[uv_offset];

    // Convert sampled I420 value to RGB before selecting the requested output channel.
    int C = static_cast<int>(Y) - 16;
    int D = static_cast<int>(U) - 128;
    int E = static_cast<int>(V) - 128;
    int C298 = 298 * C;
    int R = (C298 + 409 * E + 128) >> 8;
    int G = (C298 - 100 * D - 208 * E + 128) >> 8;
    int B = (C298 + 516 * D + 128) >> 8;

    R = rpp_min(rpp_max(R, 0), 255);
    G = rpp_min(rpp_max(G, 0), 255);
    B = rpp_min(rpp_max(B, 0), 255);

    int rgb_value = rpp_select(B, G, static_cast<short>(c == 1));
    rgb_value = rpp_select(rgb_value, R, static_cast<short>(c == 0));
    float resized_value = static_cast<float>(rgb_value) * (1.0f / 255.0f);
    float normalized = rpp_select(0.0f, resized_value, inside);
    int dst_idx = c * output_width * output_height + y * output_width + x;
    output[dst_idx] = normalized;
}

/**
 * @brief Clear one NCHW float tensor before a content-only launch fills the resized area.
 */
__global__ void zero_nchw_f32_kernel(float* output, int output_width, int output_height)
{
    // Launch geometry covers the full model tensor, so every lane performs one valid store.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    int c = blockIdx.z;
    int dst_idx = c * output_width * output_height + y * output_width + x;
    output[dst_idx] = 0.0f;
}

/**
 * @brief Resize and normalize RGB/BGR HWC input into NCHW float output.
 */
__global__ void letterbox_resize_norm_hwc_kernel(const unsigned char* input,
                                                 int input_width,
                                                 int input_height,
                                                 int channel_base,
                                                 int channel_step,
                                                 float* output,
                                                 int output_width,
                                                 int output_height,
                                                 int resized_width,
                                                 int resized_height,
                                                 int pad_left,
                                                 int pad_top)
{
    // Each lane owns one output coordinate and one channel plane.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    int rel_x = x - pad_left;
    int rel_y = y - pad_top;
    short inside = static_cast<short>((rel_x >= 0) & (rel_x < resized_width) &
                                      (rel_y >= 0) & (rel_y < resized_height));

    // Compute source nearest-neighbor coordinates in the resized content area.
    float scale_x = static_cast<float>(input_width) / static_cast<float>(resized_width);
    float scale_y = static_cast<float>(input_height) / static_cast<float>(resized_height);
    int sample_x = rpp_min(rpp_max(rel_x, 0), resized_width - 1);
    int sample_y = rpp_min(rpp_max(rel_y, 0), resized_height - 1);
    int sx = static_cast<int>((static_cast<float>(sample_x) + 0.5f) * scale_x);
    int sy = static_cast<int>((static_cast<float>(sample_y) + 0.5f) * scale_y);
    sx = rpp_min(rpp_max(sx, 0), input_width - 1);
    sy = rpp_min(rpp_max(sy, 0), input_height - 1);

    int input_channel = channel_base + c * channel_step;
    float resized_value = static_cast<float>(input[(sy * input_width + sx) * 3 + input_channel]) * (1.0f / 255.0f);
    float normalized = rpp_select(0.0f, resized_value, inside);
    int dst_idx = c * output_width * output_height + y * output_width + x;
    output[dst_idx] = normalized;
}

/**
 * @brief Resize and normalize planar RGB CHW input into NCHW float output.
 */
__global__ void letterbox_resize_norm_chw_kernel(const unsigned char* input,
                                                 int input_width,
                                                 int input_height,
                                                 float* output,
                                                 int output_width,
                                                 int output_height,
                                                 int resized_width,
                                                 int resized_height,
                                                 int pad_left,
                                                 int pad_top)
{
    // The source and destination are both planar, so channel selects the source plane directly.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    int rel_x = x - pad_left;
    int rel_y = y - pad_top;
    short inside = static_cast<short>((rel_x >= 0) & (rel_x < resized_width) &
                                      (rel_y >= 0) & (rel_y < resized_height));

    float scale_x = static_cast<float>(input_width) / static_cast<float>(resized_width);
    float scale_y = static_cast<float>(input_height) / static_cast<float>(resized_height);
    int sample_x = rpp_min(rpp_max(rel_x, 0), resized_width - 1);
    int sample_y = rpp_min(rpp_max(rel_y, 0), resized_height - 1);
    int sx = static_cast<int>((static_cast<float>(sample_x) + 0.5f) * scale_x);
    int sy = static_cast<int>((static_cast<float>(sample_y) + 0.5f) * scale_y);
    sx = rpp_min(rpp_max(sx, 0), input_width - 1);
    sy = rpp_min(rpp_max(sy, 0), input_height - 1);

    int input_plane = c * input_width * input_height;
    float resized_value = static_cast<float>(input[input_plane + sy * input_width + sx]) * (1.0f / 255.0f);
    float normalized = rpp_select(0.0f, resized_value, inside);
    int dst_idx = c * output_width * output_height + y * output_width + x;
    output[dst_idx] = normalized;
}

/**
 * @brief Launch the I420-to-NCHW preprocessing kernel for one frame.
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
                                                   float,
                                                   float pad_x,
                                                   float pad_y)
{
    // Convert letterbox padding from metadata floats to kernel integer offsets.
    int pad_left = static_cast<int>(pad_x);
    int pad_top = static_cast<int>(pad_y);

    // Split packed I420 into planes before launching the channel-wise output kernel.
    const unsigned char* y_base = static_cast<const unsigned char*>(yuv);
    const unsigned char* u_base = y_base + input_width * input_height;
    const unsigned char* v_base = u_base + input_width * input_height / 4;

    // Launch one output row per block and one grid z-slice per output channel.
    dim3 threadsPerBlock = {0};
    dim3 blocksPerGrid = {0};
    threadsPerBlock.x = output_width;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x = 1;
    blocksPerGrid.y = output_height;
    blocksPerGrid.z = 3;

    // Fast path: clear the full model tensor once, then launch resize only over the content region.
    // Padding lanes are already zero, so the I420 kernel does not spend work on letterbox borders.
    if (resized_width > 0 && resized_height > 0 && output_width <= 8191 && resized_width <= 8191) {
        zero_nchw_f32_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            output,
            output_width,
            output_height);

        threadsPerBlock.x = resized_width;
        blocksPerGrid.y = resized_height;
        float* content_output = output +
                                static_cast<size_t>(pad_top) * static_cast<size_t>(output_width) +
                                static_cast<size_t>(pad_left);

        letterbox_resize_norm_i420_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            y_base,
            u_base,
            v_base,
            input_width,
            input_height,
            content_output,
            output_width,
            output_height,
            resized_width,
            resized_height,
            0,
            0,
            1);
        return;
    }

    letterbox_resize_norm_i420_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        y_base,
        u_base,
        v_base,
        input_width,
        input_height,
        output,
        output_width,
        output_height,
        resized_width,
        resized_height,
        pad_left,
        pad_top,
        0);
}

/**
 * @brief Launch the HWC RGB/BGR-to-NCHW preprocessing kernel for one image.
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
                                                     float,
                                                     float pad_x,
                                                     float pad_y)
{
    // Convert letterbox padding from metadata floats to kernel integer offsets.
    int pad_left = static_cast<int>(pad_x);
    int pad_top = static_cast<int>(pad_y);

    // Launch one output row per block and switch channel order for BGR inputs.
    dim3 threadsPerBlock = {0};
    dim3 blocksPerGrid = {0};
    threadsPerBlock.x = output_width;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x = 1;
    blocksPerGrid.y = output_height;
    blocksPerGrid.z = 3;

    if (resized_width > 0 && resized_height > 0 && output_width <= 8191 && resized_width <= 8191) {
        zero_nchw_f32_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            output,
            output_width,
            output_height);

        threadsPerBlock.x = resized_width;
        blocksPerGrid.y = resized_height;
        float* content_output = output +
                                static_cast<size_t>(pad_top) * static_cast<size_t>(output_width) +
                                static_cast<size_t>(pad_left);

        letterbox_resize_norm_hwc_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            static_cast<const unsigned char*>(input),
            input_width,
            input_height,
            input_is_bgr ? 2 : 0,
            input_is_bgr ? -1 : 1,
            content_output,
            output_width,
            output_height,
            resized_width,
            resized_height,
            0,
            0);
        return;
    }

    letterbox_resize_norm_hwc_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        static_cast<const unsigned char*>(input),
        input_width,
        input_height,
        input_is_bgr ? 2 : 0,
        input_is_bgr ? -1 : 1,
        output,
        output_width,
        output_height,
        resized_width,
        resized_height,
        pad_left,
        pad_top);
}

/**
 * @brief Launch the planar RGB CHW-to-NCHW preprocessing kernel for one image.
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
                                                     float,
                                                     float pad_x,
                                                     float pad_y)
{
    // Convert letterbox padding from metadata floats to kernel integer offsets.
    int pad_left = static_cast<int>(pad_x);
    int pad_top = static_cast<int>(pad_y);

    // Launch one output row per block and one grid z-slice per source channel plane.
    dim3 threadsPerBlock = {0};
    dim3 blocksPerGrid = {0};
    threadsPerBlock.x = output_width;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x = 1;
    blocksPerGrid.y = output_height;
    blocksPerGrid.z = 3;

    letterbox_resize_norm_chw_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        static_cast<const unsigned char*>(input),
        input_width,
        input_height,
        output,
        output_width,
        output_height,
        resized_width,
        resized_height,
        pad_left,
        pad_top);
}
