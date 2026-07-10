/**
 * @file nmspre_kernel.cuh
 * @brief Kernel launcher declarations for YOLOv8 PreNMS tensor rearrangement.
 */
#pragma once

#include <rpp_runtime.h>

/**
 * @brief Split YOLOv8 output into boxes and class scores for OpenRT addNMS.
 *
 * @param stream Runtime stream used to launch the kernels.
 * @param input Device pointer to the raw YOLOv8 output matrix.
 * @param width Width of the raw output matrix.
 * @param height Height of the raw output matrix.
 * @param transpose Whether the input layout is transposed.
 * @param type_size Size in bytes of one input/output element.
 * @param obj_threshold Score threshold passed through plugin configuration.
 * @param nms_threshold NMS threshold passed through plugin configuration.
 * @param max_class Maximum class count passed through plugin configuration.
 * @param out_0 Device output for box channels.
 * @param out_1 Device output for class-score channels.
 */
void RearrangeMatrix(rtStream_t stream, const void* input, int width, int height, bool transpose, int type_size,
                float obj_threshold, float nms_threshold, int max_class, void *out_0, void *out_1);
