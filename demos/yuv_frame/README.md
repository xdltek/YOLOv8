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
scripts/run.sh --model /path/to/model.onnx --demo yuv
```

Run a measured benchmark for the YUV demo:

```shell
scripts/benchmark.sh --model /path/to/model.onnx --demo yuv --loop 20
```

The repository does not ship an ONNX model. Pass one explicitly, or place the optional local fallback at `weights/yolov8s.onnx`. The default I420 input is `build/assets/test_0.yuv`, its default frame size is `1280x720`, and the default output is `build/output/output_yuv.jpg`.

To use another I420 frame through the script:

```shell
scripts/run.sh --demo yuv \
  --model /path/to/model.onnx \
  --input /path/to/frame.i420 \
  --yuv-width 1280 \
  --yuv-height 720
```

After building, the binary can also be executed directly:

```shell
build/bin/yolov8_yuv_frame_demo \
  -m /path/to/model.onnx \
  -i build/assets/test_0.yuv \
  --yuv-width 1280 \
  --yuv-height 720 \
  -o build/output/output_yuv.jpg \
  -l 1
```

## Arguments

```text
-m, --model      ONNX model path. Required.
-i, --input      One-frame I420 YUV file. Default: ../assets/test_0.yuv
--yuv-width      I420 frame width. Default: 1280
--yuv-height     I420 frame height. Default: 720
-o, --output     Rendered output image path. Default: ../output/output_yuv.jpg
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

The demo prints stage banners, cache status, progress, and a performance report:

```text
[ENGINE  ] Reusing cached OpenRT engine: <cache-path>
[WARMUP  ] Running 1 unmeasured frame for stable performance
[BENCH   ] Measuring <loop> full-pipeline frames
End-to-end FPS           : <fps>
OpenRT inference FPS     : <fps>
Detections (last frame)  : <count>
Input H2D transfer       : <ms> ms  (<KiB> KiB)
Preprocess               : <ms> ms
OpenRT inference         : <ms> ms
Postprocess              : <ms> ms
Output D2H transfer      : <ms> ms  (<KiB> KiB)
Visualization            : <output-path>
```

Transfer time is reported separately. End-to-end latency and FPS use `preprocess + inference + postprocess`.

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
scripts/run.sh --model /path/to/model.onnx --demo yuv
```

执行 YUV demo benchmark：

```shell
scripts/benchmark.sh --model /path/to/model.onnx --demo yuv --loop 20
```

仓库不随附 ONNX 模型。可以显式传入外部模型，或自行把可选 fallback 放到 `weights/yolov8s.onnx`。默认 I420 输入为 `build/assets/test_0.yuv`，默认帧尺寸为 `1280x720`，默认输出为 `build/output/output_yuv.jpg`。

通过脚本使用其他 I420 输入帧：

```shell
scripts/run.sh --demo yuv \
  --model /path/to/model.onnx \
  --input /path/to/frame.i420 \
  --yuv-width 1280 \
  --yuv-height 720
```

构建完成后，也可以直接执行二进制：

```shell
build/bin/yolov8_yuv_frame_demo \
  -m /path/to/model.onnx \
  -i build/assets/test_0.yuv \
  --yuv-width 1280 \
  --yuv-height 720 \
  -o build/output/output_yuv.jpg \
  -l 1
```

## 参数

```text
-m, --model      ONNX 模型路径。必填。
-i, --input      单帧 I420 YUV 文件。默认：../assets/test_0.yuv
--yuv-width      I420 帧宽。默认：1280
--yuv-height     I420 帧高。默认：720
-o, --output     绘制检测框后的输出图片路径。默认：../output/output_yuv.jpg
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

demo 会打印阶段 banner、缓存状态、运行进度和性能报告：

```text
[ENGINE  ] Reusing cached OpenRT engine: <cache-path>
[WARMUP  ] Running 1 unmeasured frame for stable performance
[BENCH   ] Measuring <loop> full-pipeline frames
End-to-end FPS           : <fps>
OpenRT inference FPS     : <fps>
Detections (last frame)  : <count>
Input H2D transfer       : <ms> ms  (<KiB> KiB)
Preprocess               : <ms> ms
OpenRT inference         : <ms> ms
Postprocess              : <ms> ms
Output D2H transfer      : <ms> ms  (<KiB> KiB)
Visualization            : <output-path>
```

传输时间单独报告；端到端 latency 和 FPS 使用 `preprocess + inference + postprocess` 计算。

绘制后的图片默认保存到 `build/output/output_yuv.jpg`。

## 相关文档

- [项目 API 文档](../../docs/api/README.md)：可复用 API 索引。
- [RPP 加速模块](../../rpp/README.md)：RPP 执行层说明。

## 复用说明

当应用接收原始 I420 帧时，可使用本 demo。推理路径会直接使用标准 I420 顺序打包的 Y、U、V 平面调用 `YoloV8Pipeline::runI420()`。
