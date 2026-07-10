<a id="english"></a>

# YOLOv8 Inference Demo on RPP

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

![XDL Logo](docs/logo/logo_color_horizontal.png)

This repository is a C++ demo for running a YOLOv8 object detection model through OpenRT on RPP accelerator hardware.
It helps users quickly build the project, run inference with default settings, and reuse the source code for edge inference applications.

**Table Of Contents**

- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Demo Guides](#demo-guides)
- [Documentation](#documentation)
- [Prerequisites](#prerequisites)
- [How to Build](#how-to-build)
- [How to Run](#how-to-run)
- [Direct Binary Usage](#direct-binary-usage)
- [Code Execution Workflow](#code-execution-workflow)
- [Key Source Files](#key-source-files)
- [Reusable Components](#reusable-components)
- [How to Adapt This Demo](#how-to-adapt-this-demo)

## Quick Start

Run from the repository root. `scripts/build.sh` downloads and verifies the default YOLOv8s ONNX model under `weights/` when it is missing.

```shell
scripts/build.sh
scripts/run.sh
```

Successful execution should print warmup-stable stage timing and FPS for the RGB and YUV demos, and generate rendered images under `build/output`.

The standard scripts are:

| Script | Purpose |
|---|---|
| `scripts/build.sh` | Downloads the default model when needed, cleans previous build outputs, configures CMake, and builds the project. |
| `scripts/run.sh` | Runs all demos by default, or one selected demo with configurable model, input, output, and loop count. |
| `scripts/benchmark.sh` | Runs all demos by default with repeated measured loops and saves a full log automatically. |

Common first checks:

```shell
scripts/run.sh --demo rgb
scripts/run.sh --demo yuv
scripts/benchmark.sh
```

`scripts/run.sh` resolves paths from the repository root. By default it runs `rgb` and `yuv` serially, uses `weights/yolov8s.onnx`, reads inputs from `build/assets`, and writes rendered images to `build/output`. If `/usr/local/rpp/bin/rpp_server` exists, the script ensures it is running before launching the demo and writes the server log to `build/logs/rpp_server.log`.

`scripts/benchmark.sh` saves the full console log to `build/logs/benchmark_<demo>_<timestamp>.log`.

## Repository Layout

Linked paths in this section can be clicked to jump to their detailed README.

- `CMakeLists.txt`
  Root build entry. Configures shared dependencies and builds the reusable libraries and demo targets.
- `src/perf/`
  Optional `rpp_perf` trace session wrapper. It is active only when CMake is configured with `-DYOLO_ENABLE_RPP_PERF=ON`; perf-enabled demo builds automatically save trace files under `build/trace`.
- `src/utils/`
  Logging, ONNX parse helper, buffer helpers, and CLI/path utilities.
- `src/yolov8/`
  YOLOv8-specific labels, detection types, visualization, and pipeline composition.
- [`rpp/`](rpp/README.md)
  RPP hardware-facing modules, including OpenRT inference, preprocessing kernels/modules, and YOLOv8 postprocessing.
- `demos/`
  Executable demo entries, including RGB image and I420 YUV frame demos.
- [`docs/`](docs/README.md)
  Public project documentation, logo assets, and API documentation links.
- `scripts/`
  Standard build, run, and benchmark command entry points.
- `assets/`
  Source sample image and I420 frame copied into `build/assets/` for quick verification.
- `weights/`
  Local model drop-in directory. The directory is kept in Git, but model files are ignored.
- `3rd_party/`
  `argparse` header-only library used by demo main files for CLI parsing.

This repository does not track ONNX model files in Git. `scripts/build.sh` keeps `weights/yolov8s.onnx` available by downloading it from `xdltek/onnx_model_tool` and checking its MD5 when needed. Class names for overlays are the embedded MS COCO 80-class list in `src/yolov8/labels.cpp` (`coco80_class_labels()`), matching standard YOLO COCO pretrained ordering.

## Demo Guides

The root README explains the shared build and architecture. Each executable has its own guide with its input format, command-line options, workflow, and expected output:

- [RGB image demo](demos/rgb_image/README.md): RGB/BGR image demo guide.
- [I420 YUV frame demo](demos/yuv_frame/README.md): I420 YUV frame demo guide.

## Documentation

- [Project documentation](docs/README.md): public documentation entry.
- [API documentation](docs/api/README.md): reusable API index.
- [RPP acceleration modules](rpp/README.md): RPP module guide.
- [Performance trace helper](src/perf/README.md): perf trace guide.
- [Release notes](ReleaseNote.md): project release notes.

## Prerequisites

Before building this demo, make sure the following dependencies are available:

- Linux development environment
- CMake `>= 3.16.3`
- A compiler with C++17 support
- OpenCV with the components required by this project
- OpenRT installed under `/usr/local/rpp`
- Runtime libraries required by the target RPP accelerator card environment
- `wget` and `md5sum` when the default model must be downloaded by `scripts/build.sh`

If your OpenRT installation path is different, update the corresponding path settings in `CMakeLists.txt`.

## How to Build

Run the following command from the repository root:

```shell
scripts/build.sh
```

If `weights/yolov8s.onnx` is missing, the build script downloads the default model and verifies:

```shell
e86050e2e944cd8944863fd8fddd1dea  weights/yolov8s.onnx
```

After the build completes successfully, the executables should be available at:

```shell
build/bin/yolov8_rgb_image_demo
build/bin/yolov8_yuv_frame_demo
```

Build outputs are grouped as:

```text
build/bin/      demo executables
build/lib/      libyolov8_core.so, librpp_inference.so, librpp_yolo_preprocess.so, librpp_yolov8_postprocess.so, and helper libraries
build/assets/   copied sample inputs
build/output/   rendered detection images
build/trace/    optional rpp_perf traces
build/logs/     script and benchmark logs
```

## How to Run

Run demos from the repository root after building the project. The default model path is `weights/yolov8s.onnx`; pass `--model` only when using a different model.

```shell
scripts/run.sh
```

By default, `scripts/run.sh` runs the RGB and YUV demos serially. Use `--demo rgb` or `--demo yuv` when running a single demo or passing a custom input/output path.

Benchmark runs use the same default model and inputs:

```shell
scripts/benchmark.sh
scripts/benchmark.sh --demo rgb --loop 20
scripts/benchmark.sh --demo yuv --loop 20
```

Each script supports `-h` and `--help`.

## Direct Binary Usage

The scripts are the recommended first-use entry point because they resolve paths and protect the `rpp_server` startup flow. After building, users can also run the demo binaries directly from the repository root by passing explicit paths:

```shell
build/bin/yolov8_rgb_image_demo \
  -o weights/yolov8s.onnx \
  -i build/assets/test.jpg \
  --output build/output/output_rgb.jpg \
  -l 1

build/bin/yolov8_yuv_frame_demo \
  -o weights/yolov8s.onnx \
  --yuv build/assets/test_0.yuv \
  --yuv-width 1280 \
  --yuv-height 720 \
  --output build/output/output_yuv.jpg \
  -l 1
```

When running binaries directly, make sure `rpp_server` is already running if the installed RPP environment requires it.

For full command-line options and expected output, see the per-demo guides:

- [RGB image demo](demos/rgb_image/README.md): RGB/BGR image demo guide.
- [I420 YUV frame demo](demos/yuv_frame/README.md): I420 YUV frame demo guide.

## Code Execution Workflow

```text
+------------------------+
| Parse CLI arguments    |
| and resolve input paths|
+------------------------+
            |
            v
+------------------------+
| Init optional perf     |
| trace session in perf  |
| builds                 |
+------------------------+
            |
            v
+------------------------+
| Init YOLOv8 pipeline   |
| and OpenRT engine      |
+------------------------+
            |
            v
+------------------------+
| Run full pipeline once |
| as explicit warmup     |
+------------------------+
            |
            v
+------------------------+
| Run measured pipeline: |
| preprocess + infer +   |
| postprocess            |
+------------------------+
            |
            v
+------------------------+
| Draw boxes, save image,|
| print latency and FPS  |
+------------------------+
```

## Key Source Files

- `src/yolov8/yolo_pipeline.h` and `src/yolov8/yolo_pipeline.cpp`
  Reusable YOLOv8 application pipeline.
- `src/yolov8/detection_types.h`
  Shared detection, letterbox, config, and timing data structures.
- `src/yolov8/labels.cpp`
  Embedded COCO 80-class labels.
- `src/yolov8/visualization.cpp`
  Detection drawing and image output helpers.
- `rpp/inference/rpp_infer_engine.h`
  Model-agnostic OpenRT inference wrapper.
- `rpp/preprocess/rpp_yolo_preprocessor.h`
  RGB/BGR and I420 input preprocessing wrapper.
- `rpp/postprocess/rpp_yolov8_postprocessor.h`
  YOLOv8 output layout resolution, RT_NMS integration, and fallback postprocessing.

## Reusable Components

- `YoloV8Pipeline`
  Application-facing pipeline for RGB/BGR image and packed I420 frame input.
- `RppInferEngine`
  Generic OpenRT ONNX inference wrapper.
- `RppYoloPreprocessor`
  RPP preprocessing wrapper for resize, letterbox, color conversion, and normalization.
- `YoloV8Postprocessor`
  YOLOv8 decode, filtering, and NMS result conversion.
- `PerfTraceSession`
  Optional trace session helper for perf-enabled builds.

## How to Adapt This Demo

For a new YOLOv8-style application:

1. Keep the standard build/run scripts as the outer entry points.
2. Replace the model under `weights/` or pass a custom model with `--model`.
3. Confirm the model output layout is compatible with `YoloV8Postprocessor`.
4. Add input-specific loading code in a demo executable if RGB/BGR image or I420 frame input is not enough.
5. Keep model labels, visualization, and reporting in `src/yolov8/` or the demo layer, not inside the generic inference wrapper.

---

<a id="chinese"></a>

# YOLOv8 RPP 推理 Demo

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应语言章节。

![XDL Logo](docs/logo/logo_color_horizontal.png)

本仓库是一个 C++ YOLOv8 目标检测 demo，通过 OpenRT 在 RPP 加速器硬件上运行。它用于帮助用户快速构建工程、使用默认配置运行推理，并复用源码开发边缘推理应用。

**目录**

- [快速开始](#快速开始)
- [仓库结构](#仓库结构)
- [Demo 指南](#demo-指南)
- [文档导航](#文档导航)
- [环境依赖](#环境依赖)
- [如何构建](#如何构建)
- [如何运行](#如何运行)
- [直接执行二进制](#直接执行二进制)
- [代码执行流程](#代码执行流程)
- [关键源码](#关键源码)
- [可复用组件](#可复用组件)
- [如何适配此 Demo](#如何适配此-demo)

## 快速开始

从仓库根目录执行。`scripts/build.sh` 会在缺少默认模型时下载并校验 `weights/yolov8s.onnx`。

```shell
scripts/build.sh
scripts/run.sh
```

执行成功后，应打印 RGB 和 YUV demo 的阶段耗时与 FPS，并在 `build/output` 下生成绘制后的图片。

标准脚本如下：

| 脚本 | 用途 |
|---|---|
| `scripts/build.sh` | 在需要时下载默认模型，清理旧构建产物，配置 CMake，并构建工程。 |
| `scripts/run.sh` | 默认运行全部 demo，也可选择单个 demo 并配置模型、输入、输出和 loop 次数。 |
| `scripts/benchmark.sh` | 默认对全部 demo 执行多轮测量，并自动保存完整日志。 |

常用首次检查：

```shell
scripts/run.sh --demo rgb
scripts/run.sh --demo yuv
scripts/benchmark.sh
```

`scripts/run.sh` 会从仓库根目录解析路径。默认情况下，它串行运行 `rgb` 和 `yuv`，使用 `weights/yolov8s.onnx`，从 `build/assets` 读取输入，并把绘制结果写入 `build/output`。如果存在 `/usr/local/rpp/bin/rpp_server`，脚本会在启动 demo 前确保服务运行，并把服务日志写入 `build/logs/rpp_server.log`。

`scripts/benchmark.sh` 会把完整控制台日志保存到 `build/logs/benchmark_<demo>_<timestamp>.log`。

## 仓库结构

本节中的链接可跳转到对应详细 README。

- `CMakeLists.txt`
  根构建入口，配置共享依赖并构建可复用库和 demo target。
- `src/perf/`
  可选 `rpp_perf` trace session 封装。仅在 CMake 使用 `-DYOLO_ENABLE_RPP_PERF=ON` 配置时生效；perf 构建下 demo 会自动把 trace 保存到 `build/trace`。
- `src/utils/`
  日志、ONNX parse 辅助、buffer 辅助以及 CLI/路径工具。
- `src/yolov8/`
  YOLOv8 专属标签、检测类型、可视化和 pipeline 组合。
- [`rpp/`](rpp/README.md)
  面向 RPP 硬件的模块，包括 OpenRT 推理、前处理 kernel/module 和 YOLOv8 后处理。
- `demos/`
  可执行 demo 入口，包括 RGB 图片和 I420 YUV 帧 demo。
- [`docs/`](docs/README.md)
  公开项目文档、logo 资源和 API 文档链接。
- `scripts/`
  标准构建、运行和 benchmark 命令入口。
- `assets/`
  源示例图片和 I420 帧，会复制到 `build/assets/` 用于快速验证。
- `weights/`
  本地模型目录。目录会保留在 Git 中，但模型文件被忽略。
- `3rd_party/`
  demo main 使用的 `argparse` header-only 库。

本仓库不在 Git 中跟踪 ONNX 模型文件。`scripts/build.sh` 会从 `xdltek/onnx_model_tool` 下载并校验 `weights/yolov8s.onnx`。框选显示使用 `src/yolov8/labels.cpp` 中内置的 MS COCO 80 类标签表（`coco80_class_labels()`），匹配标准 YOLO COCO 预训练类别顺序。

## Demo 指南

根 README 说明共享构建方式和整体架构。每个可执行 demo 都有自己的说明，包含输入格式、命令行选项、执行流程和预期输出：

- [RGB 图片 demo](demos/rgb_image/README.md)：RGB/BGR 图片 demo 指南。
- [I420 YUV 帧 demo](demos/yuv_frame/README.md)：I420 YUV 帧 demo 指南。

## 文档导航

- [项目文档](docs/README.md)：公开文档入口。
- [API 文档](docs/api/README.md)：可复用 API 索引。
- [RPP 加速模块](rpp/README.md)：RPP 模块指南。
- [性能 trace 辅助模块](src/perf/README.md)：perf trace 指南。
- [发布说明](ReleaseNote.md)：项目发布说明。

## 环境依赖

构建前请确认环境中具备：

- Linux 开发环境
- CMake `>= 3.16.3`
- 支持 C++17 的编译器
- 工程所需 OpenCV 组件
- OpenRT 安装在 `/usr/local/rpp`
- 目标 RPP 加速卡环境所需运行时库
- 当 `scripts/build.sh` 需要下载默认模型时，需要 `wget` 和 `md5sum`

如果 OpenRT 安装路径不同，请更新 `CMakeLists.txt` 中的对应路径配置。

## 如何构建

从仓库根目录执行：

```shell
scripts/build.sh
```

如果缺少 `weights/yolov8s.onnx`，构建脚本会下载默认模型并校验：

```shell
e86050e2e944cd8944863fd8fddd1dea  weights/yolov8s.onnx
```

构建成功后，可执行文件位于：

```shell
build/bin/yolov8_rgb_image_demo
build/bin/yolov8_yuv_frame_demo
```

构建产物分组如下：

```text
build/bin/      demo 可执行文件
build/lib/      libyolov8_core.so、librpp_inference.so、librpp_yolo_preprocess.so、librpp_yolov8_postprocess.so 以及辅助库
build/assets/   复制后的示例输入
build/output/   绘制后的检测图片
build/trace/    可选 rpp_perf trace
build/logs/     脚本和 benchmark 日志
```

## 如何运行

构建完成后，从仓库根目录运行 demo。默认模型路径是 `weights/yolov8s.onnx`；只有使用其他模型时才需要传入 `--model`。

```shell
scripts/run.sh
```

默认情况下，`scripts/run.sh` 串行运行 RGB 和 YUV demo。需要单独运行一个 demo 或传入自定义输入/输出路径时，使用 `--demo rgb` 或 `--demo yuv`。

Benchmark 使用相同的默认模型和输入：

```shell
scripts/benchmark.sh
scripts/benchmark.sh --demo rgb --loop 20
scripts/benchmark.sh --demo yuv --loop 20
```

每个脚本都支持 `-h` 和 `--help`。

## 直接执行二进制

推荐首次使用标准脚本，因为脚本会解析路径并处理 `rpp_server` 启动流程。构建后，也可以从仓库根目录传入显式路径直接执行 demo 二进制：

```shell
build/bin/yolov8_rgb_image_demo \
  -o weights/yolov8s.onnx \
  -i build/assets/test.jpg \
  --output build/output/output_rgb.jpg \
  -l 1

build/bin/yolov8_yuv_frame_demo \
  -o weights/yolov8s.onnx \
  --yuv build/assets/test_0.yuv \
  --yuv-width 1280 \
  --yuv-height 720 \
  --output build/output/output_yuv.jpg \
  -l 1
```

如果直接运行二进制，请确认当前 RPP 环境需要的 `rpp_server` 已经启动。

完整命令行选项和预期输出见 demo 级文档：

- [RGB 图片 demo](demos/rgb_image/README.md)：RGB/BGR 图片 demo 指南。
- [I420 YUV 帧 demo](demos/yuv_frame/README.md)：I420 YUV 帧 demo 指南。

## 代码执行流程

```text
+------------------------+
| 解析命令行参数和路径   |
+------------------------+
            |
            v
+------------------------+
| perf 构建中初始化可选  |
| trace session          |
+------------------------+
            |
            v
+------------------------+
| 初始化 YOLOv8 pipeline |
| 和 OpenRT engine       |
+------------------------+
            |
            v
+------------------------+
| 显式执行一次完整 warmup|
+------------------------+
            |
            v
+------------------------+
| 执行测量 pipeline：    |
| preprocess + infer +   |
| postprocess            |
+------------------------+
            |
            v
+------------------------+
| 绘制检测框、保存图片、 |
| 打印 latency 和 FPS    |
+------------------------+
```

## 关键源码

- `src/yolov8/yolo_pipeline.h` 和 `src/yolov8/yolo_pipeline.cpp`
  可复用 YOLOv8 应用 pipeline。
- `src/yolov8/detection_types.h`
  共享检测结果、letterbox、配置和计时数据结构。
- `src/yolov8/labels.cpp`
  内置 COCO 80 类标签。
- `src/yolov8/visualization.cpp`
  检测框绘制和图片输出辅助。
- `rpp/inference/rpp_infer_engine.h`
  模型无关的 OpenRT 推理封装。
- `rpp/preprocess/rpp_yolo_preprocessor.h`
  RGB/BGR 和 I420 输入前处理封装。
- `rpp/postprocess/rpp_yolov8_postprocessor.h`
  YOLOv8 输出 layout 解析、RT_NMS 集成和 fallback 后处理。

## 可复用组件

- `YoloV8Pipeline`
  面向 RGB/BGR 图片和 packed I420 帧输入的应用层 pipeline。
- `RppInferEngine`
  通用 OpenRT ONNX 推理封装。
- `RppYoloPreprocessor`
  用于 resize、letterbox、颜色转换和归一化的 RPP 前处理封装。
- `YoloV8Postprocessor`
  YOLOv8 decode、过滤和 NMS 结果转换。
- `PerfTraceSession`
  perf 构建中使用的可选 trace session 辅助。

## 如何适配此 Demo

适配新的 YOLOv8 应用时：

1. 保留标准 build/run 脚本作为外部入口。
2. 替换 `weights/` 下的模型，或通过 `--model` 传入自定义模型。
3. 确认模型输出 layout 与 `YoloV8Postprocessor` 兼容。
4. 如果 RGB/BGR 图片或 I420 帧输入不满足需求，在 demo 可执行入口中添加新的输入加载逻辑。
5. 将模型标签、可视化和报告逻辑保留在 `src/yolov8/` 或 demo 层，不放入通用推理封装。
