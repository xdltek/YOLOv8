<a id="english"></a>

# RPP Preprocess Module

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

This module prepares YOLOv8 model input tensors on RPP.

## Responsibilities

- Validate host or device input metadata.
- Copy host input through a reusable pinned staging buffer.
- Compute YOLO letterbox scale and padding.
- Convert input format, resize, and normalize into FP32 NCHW model layout.
- Reuse runtime and SRAM workspaces across frames.
- Report H2D and preprocessing timings.

## Supported Formats

- `BGR_HWC`
- `RGB_HWC`
- `RGB_CHW`
- `YUV_I420`

The RGB application uses `BGR_HWC`; the raw-frame application uses `YUV_I420`. `NV12` and `NV21` remain reserved enum values and are rejected until format-specific kernels are provided.

## Main API

- `RppYoloPreprocessor::init()` configures the maximum source size and model input shape.
- `RppYoloPreprocessor::run()` writes one preprocessed frame directly into the model input device buffer.
- `RppYoloPreprocessor::releaseSramBuffers()` returns temporary SRAM before another pipeline stage needs it.
- `make_letterbox_info()` produces geometry metadata shared with postprocessing.

## Reuse Notes

Reuse this module for static FP32 NCHW inputs that require YOLO letterbox and `[0,1]` normalization. The model input dimensions are derived from the selected ONNX model, including the validated 640×640 and 1280×1024 contracts.

---

<a id="chinese"></a>

# RPP 前处理模块

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应语言章节。

本模块在 RPP 上准备 YOLOv8 模型输入 tensor。

## 模块职责

- 校验 host 或 device 输入元数据。
- 通过可复用的页锁定 Host 中转 buffer 复制输入。
- 计算 YOLO letterbox 缩放比例和 padding。
- 完成格式转换、resize、归一化，并写入 FP32 NCHW 模型输入布局。
- 跨帧复用 runtime 和 SRAM workspace。
- 上报 H2D 和前处理耗时。

## 支持格式

- `BGR_HWC`
- `RGB_HWC`
- `RGB_CHW`
- `YUV_I420`

RGB 应用使用 `BGR_HWC`，原始帧应用使用 `YUV_I420`。`NV12` 和 `NV21` 目前为预留枚举值，在提供对应 kernel 前会被拒绝。

## 主要 API

- `RppYoloPreprocessor::init()` 配置最大源图尺寸和模型输入 shape。
- `RppYoloPreprocessor::run()` 将一帧前处理结果直接写入模型输入 device buffer。
- `RppYoloPreprocessor::releaseSramBuffers()` 在其他 pipeline 阶段需要 SRAM 前释放临时空间。
- `make_letterbox_info()` 生成与后处理共享的几何信息。

## 复用说明

本模块适用于要求 YOLO letterbox 和 `[0,1]` 归一化的静态 FP32 NCHW 输入。模型输入尺寸从所选 ONNX 模型自动获取，包括已验证的 640×640 和 1280×1024 契约。
