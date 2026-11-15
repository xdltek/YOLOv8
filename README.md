<a id="english"></a>

# RPP-Accelerated YOLOv8 Application

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
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Validated Models](#validated-models)
- [Appendix: Export ONNX From a PT Model](#appendix-export-onnx-from-a-pt-model)

## Quick Start

Run from the repository root. Building does not require an ONNX model. This repository does not include or download model files; provide a supported model when running inference.

```shell
scripts/build.sh
scripts/run.sh --model /path/to/yolov8_model.onnx
```

Successful execution should print warmup-stable stage timing and FPS for the RGB and YUV demos, and generate rendered images under `build/output`.

The standard scripts are:

| Script | Purpose |
|---|---|
| `scripts/build.sh` | Cleans previous build outputs, configures CMake, and builds the project without downloading a model. |
| `scripts/run.sh` | Runs all demos by default, or one selected demo with configurable model, input, output, and loop count. |
| `scripts/benchmark.sh` | Runs all demos by default with repeated measured loops and saves a full log automatically. |

Common first checks:

```shell
scripts/run.sh --model /path/to/model.onnx --demo rgb
scripts/run.sh --model /path/to/model.onnx --demo yuv
scripts/benchmark.sh --model /path/to/model.onnx
```

`scripts/run.sh` resolves paths from the repository root and runs `rgb` and `yuv` serially unless one workload is selected. The model path is always required through `-m/--model`; no repository-local fallback is assumed. Reference inputs come from `build/assets`, and rendered images are written to `build/output`. If `/usr/local/rpp/bin/rpp_server` exists, the script ensures it is running before launch and writes its log to `build/logs/rpp_server.log`.

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
- [`rpp/preprocess/`](rpp/preprocess/README.md)
  RPP input formats, letterbox geometry, normalization, and memory-flow details.
- [`rpp/postprocess/`](rpp/postprocess/README.md)
  YOLOv8 output contracts, transpose, score reduction, and fused RPP NMS details.
- `demos/`
  Executable demo entries, including RGB image and I420 YUV frame demos.
- [`docs/`](docs/README.md)
  Public project documentation, logo assets, and API documentation links.
- `scripts/`
  Standard build, run, and benchmark command entry points.
- `assets/`
  Source sample image and I420 frame copied into `build/assets/` for quick verification.
- `weights/`
  Empty placeholder directory only. ONNX files are not part of the repository or release package.
- `3rd_party/`
  `argparse` header-only library used by demo main files for CLI parsing.

This repository does not include or track ONNX model files in Git. Building is independent of model availability and never downloads or verifies a model. At runtime, always pass an external model path with `-m/--model`; omitting it is an error. Class names for overlays use the embedded MS COCO 80-class list in `src/yolov8/labels.cpp` (`coco80_class_labels()`); classes beyond that table are displayed as `class_<id>`.

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

If your OpenRT installation path is different, update the corresponding path settings in `CMakeLists.txt`.

## How to Build

Run the following command from the repository root:

```shell
scripts/build.sh
```

The build does not require, download, or verify an ONNX model. A supported model is required only when running inference or a benchmark.

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

Run demos from the repository root after building the project. The release package does not contain an ONNX model, so `-m/--model` is required.

```shell
scripts/run.sh --model /path/to/model.onnx
```

By default, `scripts/run.sh` runs the RGB and YUV demos serially. Use `--demo rgb` or `--demo yuv` when running a single demo or passing a custom input/output path.

Benchmark runs require the same explicit model selection:

```shell
scripts/benchmark.sh --model /path/to/model.onnx
scripts/benchmark.sh --model /path/to/model.onnx --demo rgb --loop 20
scripts/benchmark.sh --model /path/to/model.onnx --demo yuv --loop 20
```

Each script supports `-h` and `--help`.

## Direct Binary Usage

The scripts are the recommended first-use entry point because they resolve paths and protect the `rpp_server` startup flow. After building, users can also run the demo binaries directly from the repository root by passing explicit paths:

```shell
build/bin/yolov8_rgb_image_demo \
  -m /path/to/yolov8_model.onnx \
  -i build/assets/test.jpg \
  -o build/output/output_rgb.jpg \
  -l 1

build/bin/yolov8_yuv_frame_demo \
  -m /path/to/yolov8_model.onnx \
  -i build/assets/test_0.yuv \
  --yuv-width 1280 \
  --yuv-height 720 \
  -o build/output/output_yuv.jpg \
  -l 1
```

When running binaries directly, make sure `rpp_server` is already running if the installed RPP environment requires it.

### OpenRT Engine Cache

The first run builds a BF16 OpenRT engine from the selected ONNX model and stores it beside the model under `.engine_cache/`. Later runs reuse that engine when the ONNX content fingerprint, OpenRT version, precision, and engine integrity checksum still match. A changed model, missing checksum, or damaged cache triggers an automatic rebuild from ONNX.

```text
/path/to/.engine_cache/
  model.onnx.openrt<version>.bf16.<model-fingerprint>.engine
  model.onnx.openrt<version>.bf16.<model-fingerprint>.engine.fnv1a
```

The model directory must be writable to persist the cache. If it is read-only, inference can still use the engine built in memory, but the next process must build it again. Engine files are generated artifacts tied to the local OpenRT/RPP environment and must not be packaged as portable model files.

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
  Fused RPP YOLOv8 transpose, BF16 conversion, box/class slicing, and NMS.

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
2. Pass the external ONNX model path with `--model`.
3. Confirm the model output layout is compatible with `YoloV8Postprocessor`.
4. Add input-specific loading code in a demo executable if RGB/BGR image or I420 frame input is not enough.
5. Keep model labels, visualization, and reporting in `src/yolov8/` or the demo layer, not inside the generic inference wrapper.

## Known Limitations

- The application supports static FP32 YOLOv8 detection bindings. Dynamic shapes and non-detection tasks require separate adaptation.
- Supported output heads contain four pixel-space `xywh` channels followed by class scores, without embedded NMS.
- The repository does not distribute, download, or verify ONNX model files.
- COCO labels are embedded. Other taxonomies render as `class_<id>` unless the application is extended with the matching label table.
- Reported performance is local validation data and requires review on the target hardware and SDK stack.

## Troubleshooting

- If no model is present, rerun with `scripts/run.sh --model /path/to/model.onnx` or the equivalent benchmark option.
- If the demo executable is missing, build first with `scripts/build.sh`.
- If the model is rejected, verify that its static input/output shapes match the contracts documented below and that embedded NMS is disabled.
- If the selected build directory was produced by another CMake generator, rerun `scripts/build.sh --build-dir <dir>`; the script cleans only generated build outputs.
- If the installed environment requires `rpp_server`, inspect `<build-dir>/logs/rpp_server.log` after a launch failure.

## Validated Models

The application was validated with three static FP32 detection contracts. Model files were local validation inputs only and are not repository or release artifacts.

| Model family | Input | Output | Classes | Opset | Status |
|---|---|---|---:|---:|---|
| YOLOv8n/s/m/l/x COCO | `[1,3,640,640]` | `[1,84,8400]` | 80 | 17 | PASS |
| YOLOv8n/s/m/l/x OIV7 | `[1,3,640,640]` | `[1,605,8400]` | 601 | 17 | PASS |
| YOLOv8n/s/m/l/x COCO | `[1,3,1280,1024]` | `[1,84,26880]` | 80 | 18 | PASS |

Validation included ONNX checker, OpenRT/RPP engine construction, one full-pipeline warmup, 20 measured RGB frames, fused RPP postprocessing, and non-empty rendered output images. The 1280×1024 family also passed the I420 YUV application smoke test.

### COCO 640×640 performance

| Model profile | Preprocess | Inference | Postprocess | End-to-end | FPS | Status |
|---|---:|---:|---:|---:|---:|---|
| YOLOv8n 640×640 | 2.219 ms | 3.491 ms | 2.557 ms | 8.266 ms | 120.981 | PASS |
| YOLOv8s 640×640 | 2.265 ms | 5.418 ms | 2.663 ms | 10.347 ms | 96.644 | PASS |
| YOLOv8m 640×640 | 2.617 ms | 14.575 ms | 3.969 ms | 21.160 ms | 47.258 | PASS |
| YOLOv8l 640×640 | 2.624 ms | 24.427 ms | 3.896 ms | 30.947 ms | 32.313 | PASS |
| YOLOv8x 640×640 | 2.238 ms | 39.632 ms | 2.991 ms | 44.861 ms | 22.291 | PASS |

### OIV7 640×640 performance

| Model profile | Preprocess | Inference | Postprocess | End-to-end | FPS | Status |
|---|---:|---:|---:|---:|---:|---|
| YOLOv8n-OIV7 640×640 | 2.533 ms | 5.133 ms | 13.485 ms | 21.151 ms | 47.279 | PASS |
| YOLOv8s-OIV7 640×640 | 2.566 ms | 6.797 ms | 13.584 ms | 22.946 ms | 43.580 | PASS |
| YOLOv8m-OIV7 640×640 | 2.625 ms | 16.367 ms | 14.333 ms | 33.325 ms | 30.007 | PASS |
| YOLOv8l-OIV7 640×640 | 2.737 ms | 26.360 ms | 14.679 ms | 43.776 ms | 22.844 | PASS |
| YOLOv8x-OIV7 640×640 | 2.935 ms | 41.703 ms | 14.608 ms | 59.247 ms | 16.879 | PASS |

### COCO 1280×1024 performance

| Model profile | Preprocess | Inference | Postprocess | End-to-end | FPS | Status |
|---|---:|---:|---:|---:|---:|---|
| YOLOv8n 1280×1024 | 4.515 ms | 12.473 ms | 6.101 ms | 23.088 ms | 43.313 | PASS |
| YOLOv8s 1280×1024 | 4.481 ms | 22.504 ms | 6.486 ms | 33.471 ms | 29.876 | PASS |
| YOLOv8m 1280×1024 | 4.596 ms | 53.078 ms | 6.806 ms | 64.480 ms | 15.509 | PASS |
| YOLOv8l 1280×1024 | 4.453 ms | 82.287 ms | 6.583 ms | 93.323 ms | 10.716 | PASS |
| YOLOv8x 1280×1024 | 4.560 ms | 137.005 ms | 6.773 ms | 148.337 ms | 6.741 | PASS |

Each result is the average of 20 measured loops after one full-pipeline warmup. End-to-end latency is `preprocess + inference + postprocess`; model initialization and visualization are excluded. These measurements describe the local RPP validation environment and are not cross-platform performance guarantees.

> **Review required:** Confirm the target accelerator, SDK, driver, firmware, clock, utilization, and power configuration before using these values in an external performance commitment.

## Appendix: Export ONNX From a PT Model

### Export requirements

Export a YOLOv8 **detection** model with the following contract:

| Setting | Required value |
|---|---|
| Input | FP32 NCHW `[1,3,H,W]`; validated at 640×640 and 1280×1024 |
| Output | FP32 `[1,4+classes,N]` or `[1,N,4+classes]` |
| Batch size | `1` |
| Input dimensions | Static |
| ONNX opset | `17` for validated 640×640 models; `18` for validated 1280×1024 models |
| Graph simplification | Enabled |
| Embedded NMS | Disabled; this application performs fused RPP NMS |

> **Important:** Do not export with `dynamic=True` or `nms=True`. Classification, segmentation, pose, oriented-box, and end-to-end NMS outputs use different contracts and require separate postprocessing.

### Install Ultralytics and export

The following procedure mirrors the YOLOv5 export workflow and works for official or custom-trained YOLOv8 `.pt` detection weights:

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
python3 -m pip install -e .
python3 -m pip install onnx onnxruntime onnxslim

yolo export \
  model=yolov8n.pt \
  format=onnx \
  imgsz=640 \
  batch=1 \
  opset=17 \
  dynamic=False \
  simplify=True \
  nms=False
```

Replace `yolov8n.pt` with `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`, or the path to a custom-trained detection checkpoint. Ultralytics writes the ONNX file beside the PT model. The file can remain outside this repository.

### Why this differs from YOLOv5

The README structure follows the YOLOv5 application, but the upstream exporters are model-family specific. Classic YOLOv5 uses `python export.py --include onnx`; YOLOv8 uses the Ultralytics `yolo export` interface. For YOLOv8, `yolo export` is preferred because it applies the current YOLOv8-specific export logic and options. Do not run a YOLOv8 checkpoint through the YOLOv5 `export.py` script. For a reproducible release, pin the Ultralytics tag or commit used for export instead of relying on an unversioned latest checkout.

### Verify the exported contract

Run the ONNX checker and print the external tensor shapes before using the model:

```bash
python3 - <<'PY'
import onnx

path = "/path/to/yolov8n.onnx"
model = onnx.load(path, load_external_data=False)
onnx.checker.check_model(model)

def shape(value):
    return [dimension.dim_value or dimension.dim_param for dimension in value.type.tensor_type.shape.dim]

print("inputs:", [(value.name, shape(value)) for value in model.graph.input])
print("outputs:", [(value.name, shape(value)) for value in model.graph.output])
print("ONNX check: PASS")
PY
```

For a standard COCO model, the expected output is `[1,84,8400]`: four pixel-space `xywh` box channels followed by 80 class-score channels. A custom model uses `[1,4+classes,8400]`. The application derives the class count from this output shape. COCO class names are displayed automatically; custom and OIV7 classes without a bundled label table are displayed as `class_<id>`.

For the validated 1280×1024 contract, export with `imgsz=1280,1024 opset=18`; the dimensions are height followed by width. Its standard COCO output is `[1,84,26880]`.

Run the exported model with:

```bash
scripts/run.sh \
  --demo rgb \
  --model /path/to/yolov8n.onnx
```

---

<a id="chinese"></a>

# RPP 加速 YOLOv8 应用

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
- [已知限制](#已知限制)
- [故障排查](#故障排查)
- [已验证模型](#已验证模型)
- [附录：从 PT 模型导出 ONNX](#附录从-pt-模型导出-onnx)

## 快速开始

从仓库根目录执行。编译不需要 ONNX 模型。本仓库不包含也不会下载模型文件；运行推理时需提供受支持的模型。

```shell
scripts/build.sh
scripts/run.sh --model /path/to/yolov8_model.onnx
```

执行成功后，应打印 RGB 和 YUV demo 的阶段耗时与 FPS，并在 `build/output` 下生成绘制后的图片。

标准脚本如下：

| 脚本 | 用途 |
|---|---|
| `scripts/build.sh` | 清理旧构建产物、配置 CMake 并构建工程，不下载模型。 |
| `scripts/run.sh` | 默认运行全部 demo，也可选择单个 demo 并配置模型、输入、输出和 loop 次数。 |
| `scripts/benchmark.sh` | 默认对全部 demo 执行多轮测量，并自动保存完整日志。 |

常用首次检查：

```shell
scripts/run.sh --model /path/to/model.onnx --demo rgb
scripts/run.sh --model /path/to/model.onnx --demo yuv
scripts/benchmark.sh --model /path/to/model.onnx
```

`scripts/run.sh` 会从仓库根目录解析路径；未显式选择 workload 时会串行运行 `rgb` 和 `yuv`。每次运行都必须通过 `-m/--model` 指定外部模型路径，不假设仓库内存在 fallback 模型。参考输入从 `build/assets` 读取，绘制结果写入 `build/output`。如果存在 `/usr/local/rpp/bin/rpp_server`，脚本会在启动前确保服务运行，并把日志写入 `build/logs/rpp_server.log`。

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
- [`rpp/preprocess/`](rpp/preprocess/README.md)
  RPP 输入格式、letterbox 几何、归一化和内存流说明。
- [`rpp/postprocess/`](rpp/postprocess/README.md)
  YOLOv8 输出契约、转置、分数归约和融合 RPP NMS 说明。
- `demos/`
  可执行 demo 入口，包括 RGB 图片和 I420 YUV 帧 demo。
- [`docs/`](docs/README.md)
  公开项目文档、logo 资源和 API 文档链接。
- `scripts/`
  标准构建、运行和 benchmark 命令入口。
- `assets/`
  源示例图片和 I420 帧，会复制到 `build/assets/` 用于快速验证。
- `weights/`
  仅为空目录占位符；仓库和发布包均不包含 ONNX 文件。
- `3rd_party/`
  demo main 使用的 `argparse` header-only 库。

本仓库不包含 ONNX 模型文件，也不在 Git 中跟踪此类文件。编译过程与模型是否存在无关，不会下载或校验模型。运行时必须通过 `-m/--model` 指定外部模型路径；省略该参数会直接报错。可视化使用 `src/yolov8/labels.cpp` 中内置的 MS COCO 80 类列表；超出该列表的类别显示为 `class_<id>`。

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

如果 OpenRT 安装路径不同，请更新 `CMakeLists.txt` 中的对应路径配置。

## 如何构建

从仓库根目录执行：

```shell
scripts/build.sh
```

编译不需要、不会下载、也不会校验 ONNX 模型。只有执行推理或 benchmark 时才需要提供受支持的模型。

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

构建完成后，从仓库根目录运行 demo。发布包不包含 ONNX 模型，因此必须通过 `-m/--model` 指定模型。

```shell
scripts/run.sh --model /path/to/model.onnx
```

默认情况下，`scripts/run.sh` 串行运行 RGB 和 YUV demo。需要单独运行一个 demo 或传入自定义输入/输出路径时，使用 `--demo rgb` 或 `--demo yuv`。

Benchmark 同样需要显式指定模型：

```shell
scripts/benchmark.sh --model /path/to/model.onnx
scripts/benchmark.sh --model /path/to/model.onnx --demo rgb --loop 20
scripts/benchmark.sh --model /path/to/model.onnx --demo yuv --loop 20
```

每个脚本都支持 `-h` 和 `--help`。

## 直接执行二进制

推荐首次使用标准脚本，因为脚本会解析路径并处理 `rpp_server` 启动流程。构建后，也可以从仓库根目录传入显式路径直接执行 demo 二进制：

```shell
build/bin/yolov8_rgb_image_demo \
  -m /path/to/yolov8_model.onnx \
  -i build/assets/test.jpg \
  -o build/output/output_rgb.jpg \
  -l 1

build/bin/yolov8_yuv_frame_demo \
  -m /path/to/yolov8_model.onnx \
  -i build/assets/test_0.yuv \
  --yuv-width 1280 \
  --yuv-height 720 \
  -o build/output/output_yuv.jpg \
  -l 1
```

如果直接运行二进制，请确认当前 RPP 环境需要的 `rpp_server` 已经启动。

### OpenRT Engine 缓存

首次运行会从指定 ONNX 模型构建 BF16 OpenRT engine，并保存到模型同级目录的 `.engine_cache/`。后续运行仅在 ONNX 内容指纹、OpenRT 版本、精度配置和 engine 完整性校验均一致时复用缓存。模型发生变化、校验文件缺失或缓存损坏时，会自动回退到 ONNX 并重新构建。

```text
/path/to/.engine_cache/
  model.onnx.openrt<version>.bf16.<model-fingerprint>.engine
  model.onnx.openrt<version>.bf16.<model-fingerprint>.engine.fnv1a
```

模型所在目录必须可写，才能持久保存缓存。如果目录只读，本次进程仍可使用内存中刚构建的 engine，但下次启动需要重新构建。Engine 是与本机 OpenRT/RPP 环境绑定的生成产物，不能作为可移植模型文件发布。

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
  融合的 RPP YOLOv8 转置、BF16 转换、框/类别切分和 NMS 后处理。

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
2. 通过 `--model` 传入外部 ONNX 模型路径。
3. 确认模型输出 layout 与 `YoloV8Postprocessor` 兼容。
4. 如果 RGB/BGR 图片或 I420 帧输入不满足需求，在 demo 可执行入口中添加新的输入加载逻辑。
5. 将模型标签、可视化和报告逻辑保留在 `src/yolov8/` 或 demo 层，不放入通用推理封装。

## 已知限制

- 应用支持静态 FP32 YOLOv8 检测 binding；动态 shape 和非检测任务需要单独适配。
- 支持的输出 head 由 4 个像素坐标 `xywh` 通道和类别分数组成，不包含内嵌 NMS。
- 仓库不分发、下载或校验 ONNX 模型文件。
- 内置 COCO 标签；其他类别体系在未扩展匹配标签表时显示为 `class_<id>`。
- 文档中的性能数据为本地验证结果，必须在目标硬件和 SDK 软件栈上复核。

## 故障排查

- 本地没有模型时，使用 `scripts/run.sh --model /path/to/model.onnx` 或对应 benchmark 参数重新运行。
- demo 可执行文件不存在时，先执行 `scripts/build.sh`。
- 模型被拒绝时，确认静态输入/输出 shape 与下文契约一致，并确认未内嵌 NMS。
- 所选 build 目录由其他 CMake generator 生成时，重新执行 `scripts/build.sh --build-dir <dir>`；脚本只清理生成的构建产物。
- 当前环境需要 `rpp_server` 时，启动失败后检查 `<build-dir>/logs/rpp_server.log`。

## 已验证模型

应用已验证三组静态 FP32 检测契约。模型文件仅作为本地验证输入，不属于仓库或发布物。

| 模型系列 | 输入 | 输出 | 类别数 | Opset | 状态 |
|---|---|---|---:|---:|---|
| YOLOv8n/s/m/l/x COCO | `[1,3,640,640]` | `[1,84,8400]` | 80 | 17 | PASS |
| YOLOv8n/s/m/l/x OIV7 | `[1,3,640,640]` | `[1,605,8400]` | 601 | 17 | PASS |
| YOLOv8n/s/m/l/x COCO | `[1,3,1280,1024]` | `[1,84,26880]` | 80 | 18 | PASS |

验证内容包括 ONNX checker、OpenRT/RPP engine 构建、1 次完整 pipeline warmup、20 帧 RGB 正式测量、融合 RPP 后处理以及非空检测结果图片输出。1280×1024 系列还通过了 I420 YUV 应用冒烟测试。

### COCO 640×640 性能

| 模型规格 | 前处理 | 推理 | 后处理 | 端到端 | FPS | 状态 |
|---|---:|---:|---:|---:|---:|---|
| YOLOv8n 640×640 | 2.219 ms | 3.491 ms | 2.557 ms | 8.266 ms | 120.981 | PASS |
| YOLOv8s 640×640 | 2.265 ms | 5.418 ms | 2.663 ms | 10.347 ms | 96.644 | PASS |
| YOLOv8m 640×640 | 2.617 ms | 14.575 ms | 3.969 ms | 21.160 ms | 47.258 | PASS |
| YOLOv8l 640×640 | 2.624 ms | 24.427 ms | 3.896 ms | 30.947 ms | 32.313 | PASS |
| YOLOv8x 640×640 | 2.238 ms | 39.632 ms | 2.991 ms | 44.861 ms | 22.291 | PASS |

### OIV7 640×640 性能

| 模型规格 | 前处理 | 推理 | 后处理 | 端到端 | FPS | 状态 |
|---|---:|---:|---:|---:|---:|---|
| YOLOv8n-OIV7 640×640 | 2.533 ms | 5.133 ms | 13.485 ms | 21.151 ms | 47.279 | PASS |
| YOLOv8s-OIV7 640×640 | 2.566 ms | 6.797 ms | 13.584 ms | 22.946 ms | 43.580 | PASS |
| YOLOv8m-OIV7 640×640 | 2.625 ms | 16.367 ms | 14.333 ms | 33.325 ms | 30.007 | PASS |
| YOLOv8l-OIV7 640×640 | 2.737 ms | 26.360 ms | 14.679 ms | 43.776 ms | 22.844 | PASS |
| YOLOv8x-OIV7 640×640 | 2.935 ms | 41.703 ms | 14.608 ms | 59.247 ms | 16.879 | PASS |

### COCO 1280×1024 性能

| 模型规格 | 前处理 | 推理 | 后处理 | 端到端 | FPS | 状态 |
|---|---:|---:|---:|---:|---:|---|
| YOLOv8n 1280×1024 | 4.515 ms | 12.473 ms | 6.101 ms | 23.088 ms | 43.313 | PASS |
| YOLOv8s 1280×1024 | 4.481 ms | 22.504 ms | 6.486 ms | 33.471 ms | 29.876 | PASS |
| YOLOv8m 1280×1024 | 4.596 ms | 53.078 ms | 6.806 ms | 64.480 ms | 15.509 | PASS |
| YOLOv8l 1280×1024 | 4.453 ms | 82.287 ms | 6.583 ms | 93.323 ms | 10.716 | PASS |
| YOLOv8x 1280×1024 | 4.560 ms | 137.005 ms | 6.773 ms | 148.337 ms | 6.741 | PASS |

每项结果均为完整 pipeline warmup 1 次后正式测量 20 次的平均值。端到端耗时为 `前处理 + 推理 + 后处理`，不包含模型初始化和可视化。这些数据仅代表本地 RPP 验证环境，不构成跨平台性能保证。

> **需要复核：** 将这些数据用于对外性能承诺前，必须确认目标加速卡、SDK、驱动、固件、频率、利用率和功耗配置。

## 附录：从 PT 模型导出 ONNX

### 导出要求

YOLOv8 **目标检测**模型必须按以下契约导出：

| 配置项 | 必需值 |
|---|---|
| 输入 | FP32 NCHW `[1,3,H,W]`；已验证 640×640 和 1280×1024 |
| 输出 | FP32 `[1,4+类别数,N]` 或 `[1,N,4+类别数]` |
| Batch size | `1` |
| 输入维度 | 静态 |
| ONNX opset | 已验证 640×640 模型使用 `17`；已验证 1280×1024 模型使用 `18` |
| 图简化 | 启用 |
| 内嵌 NMS | 禁用；本应用执行融合 RPP NMS |

> **重要：** 不要使用 `dynamic=True` 或 `nms=True`。分类、分割、姿态、旋转框以及端到端 NMS 模型具有不同的输出契约，需要单独适配后处理。

### 安装 Ultralytics 并导出

以下流程与 YOLOv5 导出方式一致，适用于官方或自定义训练的 YOLOv8 `.pt` 检测模型：

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
python3 -m pip install -e .
python3 -m pip install onnx onnxruntime onnxslim

yolo export \
  model=yolov8n.pt \
  format=onnx \
  imgsz=640 \
  batch=1 \
  opset=17 \
  dynamic=False \
  simplify=True \
  nms=False
```

可将 `yolov8n.pt` 替换为 `yolov8s.pt`、`yolov8m.pt`、`yolov8l.pt`、`yolov8x.pt`，或自定义训练检测权重的路径。Ultralytics 会在 PT 模型所在目录生成 ONNX 文件，该文件可以保留在仓库外部。

### 为什么与 YOLOv5 不同

README 的章节结构与 YOLOv5 应用保持一致，但上游导出工具与模型系列相关。经典 YOLOv5 使用 `python export.py --include onnx`；YOLOv8 使用 Ultralytics 的 `yolo export` 接口。对于 YOLOv8，优先使用 `yolo export`，因为它包含当前 YOLOv8 专属的导出逻辑和参数。不要使用 YOLOv5 的 `export.py` 导出 YOLOv8 checkpoint。准备可复现发布时，应固定用于导出的 Ultralytics tag 或 commit，不要依赖未固定版本的最新代码。

### 验证导出契约

使用模型前，运行 ONNX checker 并打印外部 tensor shape：

```bash
python3 - <<'PY'
import onnx

path = "/path/to/yolov8n.onnx"
model = onnx.load(path, load_external_data=False)
onnx.checker.check_model(model)

def shape(value):
    return [dimension.dim_value or dimension.dim_param for dimension in value.type.tensor_type.shape.dim]

print("inputs:", [(value.name, shape(value)) for value in model.graph.input])
print("outputs:", [(value.name, shape(value)) for value in model.graph.output])
print("ONNX check: PASS")
PY
```

标准 COCO 模型的预期输出为 `[1,84,8400]`：前四个通道是像素坐标 `xywh` 检测框，随后是 80 个类别分数通道。自定义模型使用 `[1,4+类别数,8400]`。应用会从该输出 shape 自动推导类别数。COCO 模型会自动显示类别名称；未内置标签表的自定义模型和 OIV7 模型显示为 `class_<id>`。

已验证的 1280×1024 契约应使用 `imgsz=1280,1024 opset=18` 导出；尺寸顺序为高、宽。标准 COCO 输出为 `[1,84,26880]`。

使用以下命令运行导出的模型：

```bash
scripts/run.sh \
  --demo rgb \
  --model /path/to/yolov8n.onnx
```
