<a id="english"></a>

# API Documentation

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

This page links to the main reusable API surfaces in the YOLOv8 RPP demo.

## Modules

Click a module path below to jump to its guide:

- [RPP acceleration modules](../../rpp/README.md): RPP execution layer guide.
- [RPP inference module](../../rpp/inference/README.md): OpenRT inference API guide.
- [YOLOv8 application module](../../src/yolov8/README.md): pipeline API guide.
- [Utility module](../../src/utils/README.md): shared utility guide.
- [Performance trace helper](../../src/perf/README.md): perf trace guide.
- [Script usage](../../README.md#quick-start): root README quick start.
- [RGB image demo](../../demos/rgb_image/README.md): RGB demo guide.
- [I420 YUV frame demo](../../demos/yuv_frame/README.md): YUV demo guide.

## Primary Interfaces

- `YoloV8Pipeline`
  Reusable application-level pipeline for RGB/BGR image input and packed I420 frame input.
- `RppInferEngine`
  OpenRT inference wrapper for ONNX model loading, engine creation, binding buffers, and execution.
- `RppYoloPreprocessor`
  RPP preprocessing module for letterbox resize, format conversion, and normalization.
- `YoloV8Postprocessor`
  YOLOv8 output layout validation and fused RPP transpose/slice/NMS execution.

Use the module README files for migration notes, input/output contracts, and API boundaries.

---

<a id="chinese"></a>

# API 文档

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应语言章节。

本页链接 YOLOv8 RPP demo 中主要可复用 API。

## 模块

点击下方模块路径可跳转到对应说明：

- [RPP 加速模块](../../rpp/README.md)：RPP 执行层说明。
- [RPP 推理模块](../../rpp/inference/README.md)：OpenRT 推理 API 说明。
- [YOLOv8 应用模块](../../src/yolov8/README.md)：pipeline API 说明。
- [工具模块](../../src/utils/README.md)：公共工具说明。
- [性能 trace 辅助模块](../../src/perf/README.md)：perf trace 说明。
- [脚本使用](../../README.md#快速开始)：根 README 快速开始。
- [RGB 图片 demo](../../demos/rgb_image/README.md)：RGB demo 说明。
- [I420 YUV 帧 demo](../../demos/yuv_frame/README.md)：YUV demo 说明。

## 主要接口

- `YoloV8Pipeline`
  面向 RGB/BGR 图片输入和 packed I420 帧输入的可复用应用层 pipeline。
- `RppInferEngine`
  基于 OpenRT 的 ONNX 模型加载、engine 创建、binding buffer 和执行封装。
- `RppYoloPreprocessor`
  用于 letterbox resize、格式转换和归一化的 RPP 前处理模块。
- `YoloV8Postprocessor`
  用于 YOLOv8 输出 layout 校验以及融合 RPP 转置、切分和 NMS 执行的后处理模块。

迁移说明、输入输出约束和 API 边界请查看各模块 README。
