<a id="english"></a>

# RPP Acceleration Modules

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

This directory contains the hardware-facing acceleration layer used by the YOLOv8 demos. It wraps OpenRT inference, RPP preprocessing kernels, and YOLOv8 postprocessing behind reusable C++ interfaces.

The code in this directory is intentionally kept separate from demo entry code and YOLOv8 business logic. Demo programs should usually call `src/yolov8/YoloV8Pipeline` instead of invoking these low-level modules directly.

Related documentation: [API documentation](../docs/api/README.md): public API index.

## Directory Layout

- [`inference/`](inference/README.md)
  Generic OpenRT inference engine wrapper. It parses the ONNX model, builds the execution engine, owns binding buffers, and measures the inference execution interval.
- [`preprocess/`](preprocess/README.md)
  YOLO input preprocessing on RPP. It provides host-side wrapper code plus RPP kernels for RGB/BGR image input and I420 YUV frame input.
- [`postprocess/`](postprocess/README.md)
  YOLOv8 output postprocessing on RPP. One fused graph transposes channel-major output, casts it to BF16, slices boxes and class scores, and runs NMS.

## Call Flow

The normal single-image pipeline calls the RPP modules in this order:

```text
+-----------------------+
| Host input image      |
| RGB/BGR or I420 YUV   |
+-----------------------+
           |
           v
+-----------------------+
| RppYoloPreprocessor   |
| H2D + resize/format   |
+-----------------------+
           |
           v
+-----------------------+
| RppInferEngine        |
| OpenRT ONNX infer     |
+-----------------------+
           |
           v
+-----------------------+
| YoloV8Postprocessor   |
| decode + NMS on RPP   |
+-----------------------+
           |
           v
+-----------------------+
| Host detections       |
| D2H compact results   |
+-----------------------+
```

## Data Movement

The RPP layer keeps intermediate tensors on the device whenever possible:

- Input H2D happens before preprocessing. RGB/BGR demos copy decoded image bytes; YUV demos copy the I420 frame bytes.
- Preprocessing output stays on the device and is consumed directly by `RppInferEngine`.
- Inference output stays on the device and is consumed directly by `YoloV8Postprocessor`.
- Output D2H copies NMS indices plus BF16 box and score tensors required to restore selected detections for drawing.

The CPU does not run YOLOv8 decode or NMS. Unsupported output layouts fail during initialization instead of silently switching to a CPU fallback.

## Postprocess Contract

`postprocess/YoloV8Postprocessor` supports static Ultralytics detection output layouts:

```text
[C, N], [N, C], [1, C, N], [1, N, C]
```

where `C = 4 + class_count`. The first four channels are `x, y, w, h`; class scores start after the box coordinates.

If a model uses a different output layout, update `YoloV8Config`, the postprocess configuration, and detection parsing together.

## Timing Ownership

Demos print stage-level timing:

```text
Input H2D -> Preprocess -> Inference -> Postprocess -> Output D2H
```

`Input H2D` and `Output D2H` are reported as separate transfer fields. The printed end-to-end latency used for FPS is the sum of `Preprocess`, `Inference`, and `Postprocess`.

The RPP modules expose these timings through their result structures. More detailed device-side events are captured by `rpp_perf` when the project is built with `YOLO_ENABLE_RPP_PERF=ON`.

## Reuse Guidance

For a new YOLOv8-style demo:

1. Use `src/yolov8/YoloV8Pipeline` as the main integration point.
2. Select the input path that matches the source format, such as `runRgb()` or `runI420()`.
3. Keep warmup and measured loop policy in the demo executable, not in the RPP module.
4. Use the RPP classes directly only for focused inference, preprocessing, or postprocessing experiments.

For a non-YOLO model:

1. Reuse `inference/RppInferEngine` if the model can be loaded through the same OpenRT ONNX flow.
2. Add model-specific preprocessing and postprocessing modules beside the current YOLO modules or under a new model-specific directory.
3. Keep model labels, visualization, and demo-specific reporting outside this `rpp/` directory.

## Module Responsibilities

`inference/RppInferEngine` should remain model-agnostic. It owns inference initialization, engine construction, input/output buffers, and OpenRT inference execution.

`preprocess/RppYoloPreprocessor` should own only input tensor preparation. It should not know about labels, output boxes, visualization, or demo command-line options.

`postprocess/YoloV8Postprocessor` should own only YOLOv8 output tensor decoding and NMS. It should return compact detection records for host-side output handling or drawing.

---

<a id="chinese"></a>

# RPP 加速模块

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应语言章节。

本目录包含 YOLOv8 demo 使用的硬件侧加速层。它通过可复用 C++ 接口封装 OpenRT 推理、RPP 前处理 kernel 和 YOLOv8 后处理。

这里的代码有意与 demo 入口和 YOLOv8 业务逻辑分离。正常情况下，demo 程序应优先调用 `src/yolov8/YoloV8Pipeline`，而不是直接调用这些底层 RPP 模块。

相关文档：[API 文档](../docs/api/README.md)：公开 API 索引。

## 目录结构

- [`inference/`](inference/README.md)
  通用 OpenRT 推理 engine 封装。它负责解析 ONNX 模型、构建执行 engine、管理 binding buffer，并测量推理执行区间。
- [`preprocess/`](preprocess/README.md)
  YOLO 输入前处理的 RPP 实现。该目录包含 host 侧封装代码，以及面向 RGB/BGR 图片输入和 I420 YUV 帧输入的 RPP kernel。
- [`postprocess/`](postprocess/README.md)
  YOLOv8 输出后处理的 RPP 实现。它解析 YOLOv8 tensor layout，准备 RT_NMS 输入，在支持时执行 NMS，并将 compact 检测结果返回 host。

## 调用流程

常规单图 pipeline 按如下顺序调用 RPP 模块：

```text
+-----------------------+
| Host 输入图片         |
| RGB/BGR 或 I420 YUV   |
+-----------------------+
           |
           v
+-----------------------+
| RppYoloPreprocessor   |
| H2D + resize/format   |
+-----------------------+
           |
           v
+-----------------------+
| RppInferEngine        |
| OpenRT ONNX 推理      |
+-----------------------+
           |
           v
+-----------------------+
| YoloV8Postprocessor   |
| RPP 上 decode + NMS   |
+-----------------------+
           |
           v
+-----------------------+
| Host 检测结果         |
| D2H compact results   |
+-----------------------+
```

## 数据搬运

RPP 层会尽可能让中间 tensor 保留在 device 侧：

- 输入 H2D 发生在前处理之前。RGB/BGR demo 复制解码后的图片字节；YUV demo 复制 I420 帧字节。
- 前处理输出保留在 device 侧，并直接作为 `RppInferEngine` 的输入。
- 推理输出保留在 device 侧，并直接作为 `YoloV8Postprocessor` 的输入。
- 输出 D2H 复制 NMS 索引，以及恢复选中检测框所需的 BF16 框和分数 tensor。

CPU 不执行 YOLOv8 decode 或 NMS。不支持的输出 layout 会在初始化阶段直接失败，不会静默切换到 CPU fallback。

## 后处理约束

`postprocess/YoloV8Postprocessor` 支持静态 Ultralytics 检测输出 layout：

```text
[C, N], [N, C], [1, C, N], [1, N, C]
```

其中 `C = 4 + class_count`。前四个通道为 `x, y, w, h`，类别分数位于 box 坐标之后。

如果模型使用不同的输出 layout，需要同步更新 `YoloV8Config`、后处理配置和检测结果解析逻辑。

## 计时归属

demo 打印阶段级耗时：

```text
Input H2D -> Preprocess -> Inference -> Postprocess -> Output D2H
```

`Input H2D` 和 `Output D2H` 作为独立传输字段打印。用于计算 FPS 的端到端耗时为 `Preprocess`、`Inference` 和 `Postprocess` 三个阶段之和。

RPP 模块通过各自的结果结构暴露这些耗时。更细粒度的 device 侧事件由 `rpp_perf` 捕获：项目需要使用 `YOLO_ENABLE_RPP_PERF=ON` 构建。

## 复用建议

对于新的 YOLOv8 风格 demo：

1. 使用 `src/yolov8/YoloV8Pipeline` 作为主要集成入口。
2. 根据输入格式选择对应接口，例如 `runRgb()` 或 `runI420()`。
3. 将 warmup 和测量 loop 策略放在 demo 可执行文件中，不放在 RPP 模块内部。
4. 只有在做推理、前处理或后处理专项实验时，才直接调用 RPP 类。

对于非 YOLO 模型：

1. 如果模型可以通过相同的 OpenRT ONNX 流程加载，可复用 `inference/RppInferEngine`。
2. 在当前 YOLO 模块旁边，或新的模型专属目录下，新增模型对应的前处理和后处理模块。
3. 将模型标签、可视化和 demo 专属报告逻辑放在 `rpp/` 目录之外。

## 模块职责

`inference/RppInferEngine` 应保持模型无关。它负责推理初始化、engine 构建、输入输出 buffer 和 OpenRT 推理执行。

`preprocess/RppYoloPreprocessor` 只负责输入 tensor 准备，不应关心标签、输出框、可视化或 demo 命令行参数。

`postprocess/YoloV8Postprocessor` 只负责 YOLOv8 输出 tensor 解码和 NMS，并向 host 输出 compact 检测记录，用于后续打印或绘制。
