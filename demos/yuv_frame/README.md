<a id="english"></a>

# I420 YUV Frame YOLOv8 Demo

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

This demo runs one I420 YUV frame through the reusable YOLOv8 RPP pipeline per measured loop. It is intended for camera or video workflows where the input frame is already available as planar YUV data.

## What This Demo Covers

- Load a YOLOv8-compatible ONNX model.
- Read one I420 frame from a raw `.yuv` file.
- Use RPP preprocessing to convert and letterbox the YUV input for YOLOv8.
- Run one explicit warmup pass.
- Run measured inference loops and print stage-level latency.
- Convert the same YUV frame to BGR on the host only for drawing the final output image.
- Automatically save an `rpp_perf` trace when the project is built with `YOLO_ENABLE_RPP_PERF=ON`.

## Build

Build from the repository root with the standard script:

```shell
scripts/build.sh
```

The executable is generated at:

```text
build/bin/yolov8_yuv_frame_demo
```

## Run

Run all demos with the default configuration:

```shell
scripts/run.sh
```

Run only the YUV demo from the repository root:

```shell
scripts/run.sh --demo yuv
```

Run a measured benchmark for the YUV demo:

```shell
scripts/benchmark.sh --demo yuv --loop 20
```

The script uses the default model at `weights/yolov8s.onnx`, the copied I420 frame at `build/assets/test_0.yuv`, default frame size `1280x720`, and the output path `build/output/output_yuv.jpg`.

To use another I420 frame through the script:

```shell
scripts/run.sh --demo yuv \
  --input /path/to/frame.i420 \
  --yuv-width 1280 \
  --yuv-height 720
```

After building, the binary can also be executed directly:

```shell
build/bin/yolov8_yuv_frame_demo \
  -o weights/yolov8s.onnx \
  --yuv build/assets/test_0.yuv \
  --yuv-width 1280 \
  --yuv-height 720 \
  --output build/output/output_yuv.jpg \
  -l 1
```

## Arguments

```text
-o, --onnx       ONNX model path. Default: ../../weights/yolov8s.onnx
--yuv            One-frame I420 YUV file. Default: ../assets/test_0.yuv
--yuv-width      I420 frame width. Default: 1280
--yuv-height     I420 frame height. Default: 720
--output         Rendered output image path. Default: ../output/output_yuv.jpg
-l, --loop       Measured loop count after one warmup pass. Default: 1
-v, --verbose    Enable verbose runtime logging.
```

## Execution Flow

```text
Read CLI and paths
        |
        v
Read one I420 frame from disk
        |
        v
Create BGR visualization frame on host
        |
        v
Create perf trace session in perf-enabled builds
        |
        v
Initialize YoloV8Pipeline and RppInferEngine
        |
        v
Warmup: run full pipeline once
        |
        v
Printed stages: H2D, YUV preprocess, inference, postprocess, D2H
        |
        v
Draw boxes and save rendered image
```

## Output

The demo prints model and input paths first, then a timing summary:

```text
Model path: ../../weights/yolov8s.onnx
Inference YUV path: ../assets/test_0.yuv

Summary:
Loop count: <loop>
Loop time: average per measured loop after warmup
Input H2D: <ms> ms, <bytes> bytes, <KiB> KiB
Preprocess: <ms> ms
Inference: <ms> ms
Postprocess: <ms> ms
Output D2H: <ms> ms, <bytes> bytes, <KiB> KiB
All time end to end: <ms> ms (preprocess+inference+postprocess)
FPS: <fps>
Output path: ../output/output_yuv.jpg
```

`Input H2D` and `Output D2H` are printed separately. `All time end to end` is the sum of the printed `Preprocess`, `Inference`, and `Postprocess` stages and is used for FPS.

The rendered image is saved to `build/output/output_yuv.jpg` by default.

## Related Documents

- [Project API documentation](../../docs/api/README.md): reusable API index.
- [RPP acceleration modules](../../rpp/README.md): RPP execution layer guide.

## Reuse Notes

Use this demo when your application receives raw I420 frames. The inference path calls `YoloV8Pipeline::runI420()` directly with the Y, U, and V planes packed in standard I420 order.

---

<a id="chinese"></a>

# I420 YUV 帧 YOLOv8 Demo

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应语言章节。

本 demo 在每次测量 loop 中，将一帧 I420 YUV 数据送入可复用的 YOLOv8 RPP pipeline。它适用于摄像头或视频流场景，此类场景中输入帧通常已经是 planar YUV 格式。

## 这个 Demo 包含什么

- 加载 YOLOv8 兼容 ONNX 模型。
- 从原始 `.yuv` 文件读取一帧 I420 数据。
- 使用 RPP 前处理完成 YUV 输入转换和 YOLOv8 letterbox。
- 显式执行一次完整 warmup。
- 执行指定次数的测量 loop，并打印阶段级耗时。
- 仅为了最终画框，在 host 上将同一帧 YUV 转为 BGR 可视化图片。
- 在开启 `YOLO_ENABLE_RPP_PERF=ON` 构建时，自动保存 `rpp_perf` trace。

## 构建

从仓库根目录使用标准脚本构建：

```shell
scripts/build.sh
```

可执行文件生成路径：

```text
build/bin/yolov8_yuv_frame_demo
```

## 运行

使用默认配置运行全部 demo：

```shell
scripts/run.sh
```

从仓库根目录只运行 YUV demo：

```shell
scripts/run.sh --demo yuv
```

执行 YUV demo benchmark：

```shell
scripts/benchmark.sh --demo yuv --loop 20
```

脚本默认使用 `weights/yolov8s.onnx` 模型、`build/assets/test_0.yuv` I420 输入帧、`1280x720` 默认帧尺寸，并将结果保存到 `build/output/output_yuv.jpg`。

通过脚本使用其他 I420 输入帧：

```shell
scripts/run.sh --demo yuv \
  --input /path/to/frame.i420 \
  --yuv-width 1280 \
  --yuv-height 720
```

构建完成后，也可以直接执行二进制：

```shell
build/bin/yolov8_yuv_frame_demo \
  -o weights/yolov8s.onnx \
  --yuv build/assets/test_0.yuv \
  --yuv-width 1280 \
  --yuv-height 720 \
  --output build/output/output_yuv.jpg \
  -l 1
```

## 参数

```text
-o, --onnx       ONNX 模型路径。默认：../../weights/yolov8s.onnx
--yuv            单帧 I420 YUV 文件。默认：../assets/test_0.yuv
--yuv-width      I420 帧宽。默认：1280
--yuv-height     I420 帧高。默认：720
--output         绘制检测框后的输出图片路径。默认：../output/output_yuv.jpg
-l, --loop       warmup 之后的测量 loop 次数。默认：1
-v, --verbose    开启更详细的运行时日志。
```

## 执行流程

```text
读取命令行参数和路径
        |
        v
从磁盘读取一帧 I420 数据
        |
        v
在 host 上创建 BGR 可视化帧
        |
        v
perf 构建中创建 trace 会话
        |
        v
初始化 YoloV8Pipeline 和 RppInferEngine
        |
        v
Warmup：完整执行一次 pipeline
        |
        v
打印阶段：H2D、YUV preprocess、inference、postprocess、D2H
        |
        v
绘制检测框并保存结果图片
```

## 输出

demo 会先打印模型和输入路径，然后打印计时摘要：

```text
Model path: ../../weights/yolov8s.onnx
Inference YUV path: ../assets/test_0.yuv

Summary:
Loop count: <loop>
Loop time: average per measured loop after warmup
Input H2D: <ms> ms, <bytes> bytes, <KiB> KiB
Preprocess: <ms> ms
Inference: <ms> ms
Postprocess: <ms> ms
Output D2H: <ms> ms, <bytes> bytes, <KiB> KiB
All time end to end: <ms> ms (preprocess+inference+postprocess)
FPS: <fps>
Output path: ../output/output_yuv.jpg
```

`Input H2D` 和 `Output D2H` 单独打印。`All time end to end` 为打印的 `Preprocess`、`Inference` 和 `Postprocess` 三个阶段之和，并用于计算 FPS。

绘制后的图片默认保存到 `build/output/output_yuv.jpg`。

## 相关文档

- [项目 API 文档](../../docs/api/README.md)：可复用 API 索引。
- [RPP 加速模块](../../rpp/README.md)：RPP 执行层说明。

## 复用说明

当应用接收原始 I420 帧时，可使用本 demo。推理路径会直接使用标准 I420 顺序打包的 Y、U、V 平面调用 `YoloV8Pipeline::runI420()`。
