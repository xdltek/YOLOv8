# Release Notes

## YOLOv8 RPP Inference Demo V2.0

| Item | Value |
|---|---|
| Release version | V2.0 |
| Target code | Current repository code |
| Validation | `cmake --build build -j8` passed |

### Added

- Added modular OpenRT/RPP execution components:
  - `rpp/inference/RppInferEngine` for ONNX parsing, OpenRT engine creation, binding buffer ownership, and synchronous inference.
  - `rpp/preprocess/RppYoloPreprocessor` and RPP kernels for RGB/BGR image and I420 YUV frame preprocessing.
  - `rpp/postprocess/YoloV8Postprocessor` for YOLOv8 output layout resolution, NMSPREPlugin, OpenRT `addNMS`, and CPU fallback.
- Added reusable YOLOv8 application components under `src/yolov8/`.
- Added shared utility components under `src/utils/`.
- Added optional `src/perf/PerfTraceSession` for `rpp_perf` trace capture.
- Added standard scripts under `scripts/` for build, run, and benchmark workflows.
- Added public documentation entries under `docs/` and module README files.
- Added `weights/.gitkeep` so the local model directory exists while ONNX files remain ignored by Git.
- Added detailed README coverage for the root project, demos, RPP modules, OpenRT inference, YOLOv8 application layer, utilities, performance trace helper, and API index.

### Changed

- Standardized release history as `ReleaseNote.md`.
- Standardized reusable helpers under `src/utils/`.
- Standardized the inference wrapper under `rpp/inference/`.
- Moved logo assets to `docs/logo/`.
- Moved the sample RGB image to `assets/test.jpg`.
- Updated CMake to build `rpp_inference` and use the V2.0 directory layout.
- Updated model handling so the default model path is `weights/yolov8s.onnx`; scripts can download and verify it by MD5.
- Updated the default user workflow so `scripts/build.sh`, `scripts/run.sh`, and `scripts/benchmark.sh` can be used without mandatory model or input arguments.
- Updated demo binaries to provide default model, input, output, and loop settings while still supporting explicit CLI overrides.
- Updated public documentation to use `ReleaseNote.md`, OpenRT naming, and current YOLOv8 module names consistently.
- Updated `.gitignore` coverage for build outputs, generated runtime artifacts, local ONNX/model files, and ignored internal documents.

### Removed

- Removed the root `build.sh` entry in favor of `scripts/build.sh`.
- Removed the old `doc/` layout after moving public documentation under `docs/`.
- Removed internal-document generation requirements from the public project flow.

---

## YOLOv8 RPP 推理 Demo V2.0

| 项目 | 内容 |
|---|---|
| 发布版本 | V2.0 |
| 目标代码 | 当前仓库代码 |
| 验证状态 | `cmake --build build -j8` 已通过 |

### 新增

- 新增模块化 OpenRT/RPP 执行组件：
  - `rpp/inference/RppInferEngine`：负责 ONNX 解析、OpenRT engine 创建、binding buffer 管理和同步推理。
  - `rpp/preprocess/RppYoloPreprocessor` 及 RPP kernel：负责 RGB/BGR 图片和 I420 YUV 帧前处理。
  - `rpp/postprocess/YoloV8Postprocessor`：负责 YOLOv8 输出 layout 解析、NMSPREPlugin、OpenRT `addNMS` 和 CPU fallback。
- 新增 `src/yolov8/` 可复用 YOLOv8 应用组件。
- 新增 `src/utils/` 公共工具组件。
- 新增 `src/perf/PerfTraceSession`，用于采集可选 `rpp_perf` trace。
- 新增 `scripts/` 标准构建、运行和 benchmark 入口。
- 新增 `docs/` 公开文档入口和模块 README。
- 新增 `weights/.gitkeep`，保留本地模型目录，同时继续忽略 ONNX 模型文件。
- 新增根目录、demo、RPP 模块、OpenRT 推理、YOLOv8 应用层、工具、性能 trace 辅助和 API 索引的详细 README 覆盖。

### 变更

- 发布记录统一命名为 `ReleaseNote.md`。
- 可复用 helper 统一放在 `src/utils/`。
- 推理封装统一放在 `rpp/inference/`。
- logo 资源移动到 `docs/logo/`。
- RGB 示例图片移动到 `assets/test.jpg`。
- CMake 更新为构建 `rpp_inference` 并使用 V2.0 目录布局。
- 默认模型路径更新为 `weights/yolov8s.onnx`；脚本可下载并校验 MD5。
- 默认用户流程更新为 `scripts/build.sh`、`scripts/run.sh` 和 `scripts/benchmark.sh` 均可在不强制传入模型或输入参数的情况下使用。
- demo 二进制更新为提供默认模型、输入、输出和 loop 配置，同时仍支持显式 CLI 覆盖。
- 公开文档统一使用 `ReleaseNote.md`、OpenRT 命名和当前 YOLOv8 模块名称。
- `.gitignore` 更新为覆盖构建产物、生成的运行产物、本地 ONNX/模型文件以及被忽略的内部文档。

### 移除

- 移除根目录 `build.sh`，统一使用 `scripts/build.sh`。
- 移除旧 `doc/` 布局，公开文档统一放在 `docs/`。
- 从公开项目流程中移除内部文档生成要求。
