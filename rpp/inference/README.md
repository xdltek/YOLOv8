<a id="english"></a>

# RPP Inference Module

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

This module wraps OpenRT inference for YOLOv8-compatible ONNX models.

## Responsibilities

- Parse the ONNX model.
- Build the OpenRT execution engine.
- Own model input and output binding buffers.
- Execute synchronous inference.
- Expose model input shape, output shape, and tensor data types to the pipeline.

## Main API

- `RppInferEngine::init()`
  Builds the inference engine and allocates bindings.
- `RppInferEngine::execute()`
  Runs one inference pass using the current input binding.
- `RppInferEngine::getInputDeviceBuffer()`
  Returns the device buffer consumed by preprocessing.
- `RppInferEngine::getOutputDeviceBuffer()`
  Returns the device buffer consumed by postprocessing.

## Input And Output

- Input: YOLOv8-compatible ONNX model path.
- Input binding: FP32 model input tensor.
- Output binding: FP32 YOLOv8 output tensor.

## Main Files

- `rpp_infer_engine.h`
- `rpp_infer_engine.cpp`

## Reuse Notes

Reuse this module when the model can be loaded through the same OpenRT ONNX flow. Keep model-specific preprocessing, postprocessing, labels, and visualization outside this module.

---

<a id="chinese"></a>

# RPP 推理模块

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应语言章节。

本模块封装 YOLOv8 兼容 ONNX 模型在 OpenRT 上的推理流程。

## 模块职责

- 解析 ONNX 模型。
- 构建 OpenRT 执行 engine。
- 管理模型输入和输出 binding buffer。
- 执行同步推理。
- 向 pipeline 暴露模型输入尺寸、输出尺寸和 tensor 数据类型。

## 主要 API

- `RppInferEngine::init()`
  构建推理 engine 并分配 binding。
- `RppInferEngine::execute()`
  使用当前输入 binding 执行一次推理。
- `RppInferEngine::getInputDeviceBuffer()`
  返回前处理写入的 device buffer。
- `RppInferEngine::getOutputDeviceBuffer()`
  返回后处理读取的 device buffer。

## 输入输出

- 输入：YOLOv8 兼容 ONNX 模型路径。
- 输入 binding：FP32 模型输入 tensor。
- 输出 binding：FP32 YOLOv8 输出 tensor。

## 主要文件

- `rpp_infer_engine.h`
- `rpp_infer_engine.cpp`

## 复用说明

当模型可以通过相同的 OpenRT ONNX 流程加载时，可复用本模块。模型相关的前处理、后处理、标签和可视化逻辑应放在本模块之外。
