<a id="english"></a>

# RGB Image YOLOv8 Demo

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

This demo runs one RGB/BGR image through the reusable YOLOv8 RPP pipeline per measured loop. It is a simple entry point for validating model loading, host-to-device input transfer, RPP preprocessing, OpenRT inference, YOLOv8 postprocessing, device-to-host result transfer, and rendered detection output.

## What This Demo Covers

- Load a YOLOv8-compatible ONNX model.
- Read an image with OpenCV.
- Run one explicit warmup pass.
- Run measured inference loops and print stage-level latency.
- Draw detection boxes and save the output image.
- Automatically save an `rpp_perf` trace when the project is built with `YOLO_ENABLE_RPP_PERF=ON`.

## Build

Build from the repository root with the standard script:

```shell
scripts/build.sh
```

The executable is generated at:

```text
build/bin/yolov8_rgb_image_demo
```

## Run

Run all demos with the default configuration:

```shell
scripts/run.sh
```

Run only the RGB demo from the repository root:

```shell
scripts/run.sh --model /path/to/model.onnx --demo rgb
```

Run a measured benchmark for the RGB demo:

```shell
scripts/benchmark.sh --model /path/to/model.onnx --demo rgb --loop 20
```

The repository does not ship an ONNX model. Pass one explicitly, or place the optional local fallback at `weights/yolov8s.onnx`. The default input image is `build/assets/test.jpg`, and the default output path is `build/output/output_rgb.jpg`.

To use another image through the script:

```shell
scripts/run.sh --model /path/to/model.onnx --demo rgb --input /path/to/image.jpg
```

After building, the binary can also be executed directly:

```shell
build/bin/yolov8_rgb_image_demo \
  -m /path/to/model.onnx \
  -i build/assets/test.jpg \
  -o build/output/output_rgb.jpg \
  -l 1
```

## Arguments

```text
-m, --model     ONNX model path. Required.
-i, --input     RGB/BGR image path. Default: ../assets/test.jpg
-o, --output    Rendered output image path. Default: ../output/output_rgb.jpg
-l, --loop      Measured loop count after one warmup pass. Default: 1
-v, --verbose   Enable verbose runtime logging.
```

## Execution Flow

```text
Read CLI and paths
        |
        v
Load RGB/BGR image with OpenCV
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
Printed stages: H2D, preprocess, inference, postprocess, D2H
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

The rendered image is saved to `build/output/output_rgb.jpg` by default.

## Related Documents

- [Project API documentation](../../docs/api/README.md): reusable API index.
- [RPP acceleration modules](../../rpp/README.md): RPP execution layer guide.

## Reuse Notes

Use this demo when your application already receives compressed or decoded image files on the host. For new applications, copy the CLI pattern and call `YoloV8Pipeline::runRgb()` with your own `cv::Mat`.

---

<a id="chinese"></a>

# RGB 图片 YOLOv8 Demo

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应语言章节。

本 demo 在每次测量 loop 中，将一张 RGB/BGR 图片送入可复用的 YOLOv8 RPP pipeline。它是一个简单的执行入口，可用于验证模型加载、输入 H2D、RPP 前处理、OpenRT 推理、YOLOv8 后处理、结果 D2H 以及检测框绘制输出。

## 这个 Demo 包含什么

- 加载 YOLOv8 兼容 ONNX 模型。
- 使用 OpenCV 读取图片。
- 显式执行一次完整 warmup。
- 执行指定次数的测量 loop，并打印阶段级耗时。
- 绘制检测框并保存输出图片。
- 在开启 `YOLO_ENABLE_RPP_PERF=ON` 构建时，自动保存 `rpp_perf` trace。

## 构建

从仓库根目录使用标准脚本构建：

```shell
scripts/build.sh
```

可执行文件生成路径：

```text
build/bin/yolov8_rgb_image_demo
```

## 运行

使用默认配置运行全部 demo：

```shell
scripts/run.sh
```

从仓库根目录只运行 RGB demo：

```shell
scripts/run.sh --model /path/to/model.onnx --demo rgb
```

执行 RGB demo benchmark：

```shell
scripts/benchmark.sh --model /path/to/model.onnx --demo rgb --loop 20
```

仓库不随附 ONNX 模型。可以显式传入外部模型，或自行把可选 fallback 放到 `weights/yolov8s.onnx`。默认输入图片为 `build/assets/test.jpg`，默认输出路径为 `build/output/output_rgb.jpg`。

通过脚本使用其他图片：

```shell
scripts/run.sh --model /path/to/model.onnx --demo rgb --input /path/to/image.jpg
```

构建完成后，也可以直接执行二进制：

```shell
build/bin/yolov8_rgb_image_demo \
  -m /path/to/model.onnx \
  -i build/assets/test.jpg \
  -o build/output/output_rgb.jpg \
  -l 1
```

## 参数

```text
-m, --model     ONNX 模型路径。必填。
-i, --input     RGB/BGR 图片路径。默认：../assets/test.jpg
-o, --output    绘制检测框后的输出图片路径。默认：../output/output_rgb.jpg
-l, --loop      warmup 之后的测量 loop 次数。默认：1
-v, --verbose   开启更详细的运行时日志。
```

## 执行流程

```text
读取命令行参数和路径
        |
        v
使用 OpenCV 加载 RGB/BGR 图片
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
打印阶段：H2D、preprocess、inference、postprocess、D2H
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

绘制后的图片默认保存到 `build/output/output_rgb.jpg`。

## 相关文档

- [项目 API 文档](../../docs/api/README.md)：可复用 API 索引。
- [RPP 加速模块](../../rpp/README.md)：RPP 执行层说明。

## 复用说明

当应用已经在 host 侧收到压缩图片或解码后的图片时，可使用本 demo。新应用可以复用 CLI 组织方式，并用自己的 `cv::Mat` 调用 `YoloV8Pipeline::runRgb()`。
