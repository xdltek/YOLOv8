/**
 * @file nmspre_kernel.cu
 * @brief RPP kernels that split YOLOv8 output into box and score tensors.
 */
#include "nmspre_kernel.cuh"

#include <__clang_cuda_builtin_vars.h>
#include <rpp_drv_api.h>
#include <rpp_math.h>

#include <algorithm>

/**
 * @brief Copy a channel slice from a non-transposed output matrix.
 */
__global__ void nms_arrange_trans_dw(const uint32_t* input, uint32_t* out, int width, int height, int start)
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int j   = blockIdx.y * blockDim.y + threadIdx.y;
    int idx0 = i * width + j + start;
    int idx1 = j + i * height;
    out[idx1] = input[idx0];
}

/**
 * @brief Copy a channel slice from a transposed output matrix.
 */
__global__ void nms_arrange_trans_dw_t(const uint32_t* input, uint32_t* out, int width, int height, int start)
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int j   = blockIdx.y * blockDim.y + threadIdx.y;
    int idx0 = i + (j + start) * width;
    int idx1 = j + i * height;
    out[idx1] = input[idx0];
}

void RearrangeMatrix(rtStream_t stream, const void* input, int width, int height, bool transpose, int type_size, float obj_threshold, float nms_threshold, int max_class, void *out_0, void *out_1)
{
    const int max_thread_number = 8191;
    int thread_x = width;
    int thread_y = height;
    if(transpose)
    {
        thread_x = height;
        thread_y = width;
    }
    int front_size = 4;
    int back_size = thread_x - 4;

    if(transpose)
    {
        dim3 threadsPerBlock_t(thread_y/24,  1, 1);
        dim3 blocksPerGrid_t(24, front_size, 1);
        nms_arrange_trans_dw_t<<<blocksPerGrid_t, threadsPerBlock_t, 0, stream>>>((const uint32_t*)input, (uint32_t*)out_0, width, front_size, 0);
        dim3 threadsPerBlock1_t(thread_y/24,  1, 1);
        dim3 blocksPerGrid1_t(24, back_size, 1);
        nms_arrange_trans_dw_t<<<blocksPerGrid1_t, threadsPerBlock1_t, 0, stream>>>((const uint32_t*)input, (uint32_t*)out_1, width, back_size, front_size);
    }
    else{
        dim3 threadsPerBlock(thread_y/4,  1, 1);
        dim3 blocksPerGrid(4, front_size, 1);
        nms_arrange_trans_dw<<<blocksPerGrid, threadsPerBlock, 0, stream>>>((const uint32_t*)input, (uint32_t*)out_0, width, front_size, 0);
        dim3 threadsPerBlock1(thread_y/4,  1, 1);
        dim3 blocksPerGrid1(4, back_size, 1);
        nms_arrange_trans_dw<<<blocksPerGrid1, threadsPerBlock1, 0, stream>>>((const uint32_t*)input, (uint32_t*)out_1, width, back_size, front_size);
    }
}
    
