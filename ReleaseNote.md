# Release Notes

## RPP-Accelerated YOLOv8 Application V2.0

| Item | Value |
|---|---|
| Release version | V2.0 |
| Target code | Current repository code |
| Validation | Clean build and RGB/I420 YUV functional validation passed |

### Validated Models

Fifteen static FP32 detection profiles were validated: COCO and OIV7 YOLOv8n/s/m/l/x at 640×640, plus COCO YOLOv8n/s/m/l/x at 1280×1024. RGB application validation passed for all profiles; the 1280×1024 family also passed the I420 YUV application.

Complete 640×640 and 1280×1024 performance tables are published in `README.md`. The table below summarizes the 1280×1024 COCO measurements.

| Model | End-to-end latency | FPS | Status |
|---|---:|---:|---|
| YOLOv8n | 23.088 ms | 43.313 | PASS |
| YOLOv8s | 33.471 ms | 29.876 | PASS |
| YOLOv8m | 64.480 ms | 15.509 | PASS |
| YOLOv8l | 93.323 ms | 10.716 | PASS |
| YOLOv8x | 148.337 ms | 6.741 | PASS |

These measurements are reference results from the validated RPP environment. Performance may vary with the accelerator, SDK, driver, firmware, and operating conditions.

### Added

- Added modular OpenRT/RPP execution components:
  - `rpp/inference/RppInferEngine` for ONNX parsing, OpenRT engine creation, binding buffer ownership, and synchronous inference.
  - `rpp/preprocess/RppYoloPreprocessor` and RPP kernels for RGB/BGR image and I420 YUV frame preprocessing.
  - `rpp/postprocess/YoloV8Postprocessor` for a fused RPP transpose, BF16 cast, box/class slicing, and OpenRT `addNMS` graph.
- Added reusable YOLOv8 application components under `src/yolov8/`.
- Added shared utility components under `src/utils/`.
- Added optional `src/perf/PerfTraceSession` for `rpp_perf` trace capture.
- Added standard scripts under `scripts/` for build, run, and benchmark workflows.
- Added public documentation entries under `docs/` and module README files.
- Added `weights/.gitkeep` so the local model directory exists while ONNX files remain ignored by Git.
- Added detailed README coverage for the root project, demos, RPP modules, OpenRT inference, YOLOv8 application layer, utilities, performance trace helper, and API index.
- Added dedicated bilingual preprocess and postprocess module README files to match the YOLOv5 package structure.
- Added in-memory BF16 engine initialization from the complete ONNX graph. One initialized pipeline reuses the same engine for all frames processed during its lifetime.
- Added YOLOv5-style stage banners, progress display, performance reports, and separate end-to-end/OpenRT inference FPS output.

### Changed

- Standardized release history as `ReleaseNote.md`.
- Standardized reusable helpers under `src/utils/`.
- Standardized the inference wrapper under `rpp/inference/`.
- Moved logo assets to `docs/logo/`.
- Moved the sample RGB image to `assets/test.jpg`.
- Updated CMake to build `rpp_inference` and use the V2.0 directory layout.
- Decoupled building from model availability: `scripts/build.sh` no longer downloads or verifies ONNX files.
- Updated run and benchmark workflows to accept an external ONNX path through `-m/--model`, with an optional user-provided `weights/yolov8s.onnx` fallback matching YOLOv5 behavior; the release itself still contains no model.
- Updated demo binary options to match the YOLOv5 convention: required `-m/--model`, `-i/--input`, `-o/--output`, and configurable loop count.
- Updated inference initialization to build directly from the supplied ONNX model. The application does not create or load persistent engine files.
- Updated public documentation to use `ReleaseNote.md`, OpenRT naming, and current YOLOv8 module names consistently.
- Updated `.gitignore` coverage for build outputs, generated runtime artifacts, and user-provided ONNX/model files.
- Added validated 1280×1024 YOLOv8n/s/m/l/x model contracts and measured RGB performance.

### Fixed

- Corrected I420 YUV detection with the validated 1280×1024 YOLOv8 output contract.
- Ensured RGB and I420 YUV workloads use the complete original ONNX graph without modifying the customer model.
- Ensured an initialized `YoloV8Pipeline` reuses its in-memory engine across warmup and measured frames.

### Removed

- Removed the root `build.sh` entry in favor of `scripts/build.sh`.
- Removed the old `doc/` layout after moving public documentation under `docs/`.

---

## RPP 加速 YOLOv8 应用 V2.0

| 项目 | 内容 |
|---|---|
| 发布版本 | V2.0 |
| 目标代码 | 当前仓库代码 |
| 验证状态 | 干净构建以及 RGB/I420 YUV 功能验证已通过 |

### 已验证模型

共 15 个静态 FP32 检测规格通过验证：640×640 下的 COCO 和 OIV7 YOLOv8n/s/m/l/x，以及 1280×1024 下的 COCO YOLOv8n/s/m/l/x。全部规格均通过 RGB 应用验证；1280×1024 系列同时通过 I420 YUV 应用验证。

完整的 640×640 和 1280×1024 性能表见 `README.md`。下表汇总 1280×1024 COCO 测量结果。

| 模型 | 端到端耗时 | FPS | 状态 |
|---|---:|---:|---|
| YOLOv8n | 23.088 ms | 43.313 | PASS |
| YOLOv8s | 33.471 ms | 29.876 | PASS |
| YOLOv8m | 64.480 ms | 15.509 | PASS |
| YOLOv8l | 93.323 ms | 10.716 | PASS |
| YOLOv8x | 148.337 ms | 6.741 | PASS |

这些数据是已验证 RPP 环境下的参考结果；实际性能可能随加速卡、SDK、驱动、固件和运行条件变化。

### 新增

- 新增模块化 OpenRT/RPP 执行组件：
  - `rpp/inference/RppInferEngine`：负责 ONNX 解析、OpenRT engine 创建、binding buffer 管理和同步推理。
  - `rpp/preprocess/RppYoloPreprocessor` 及 RPP kernel：负责 RGB/BGR 图片和 I420 YUV 帧前处理。
  - `rpp/postprocess/YoloV8Postprocessor`：负责融合的 RPP 转置、BF16 转换、框/类别切分和 OpenRT `addNMS` 图。
- 新增 `src/yolov8/` 可复用 YOLOv8 应用组件。
- 新增 `src/utils/` 公共工具组件。
- 新增 `src/perf/PerfTraceSession`，用于采集可选 `rpp_perf` trace。
- 新增 `scripts/` 标准构建、运行和 benchmark 入口。
- 新增 `docs/` 公开文档入口和模块 README。
- 新增 `weights/.gitkeep`，保留本地模型目录，同时继续忽略 ONNX 模型文件。
- 新增根目录、demo、RPP 模块、OpenRT 推理、YOLOv8 应用层、工具、性能 trace 辅助和 API 索引的详细 README 覆盖。
- 新增独立的双语前处理和后处理模块 README，与 YOLOv5 软件包结构保持一致。
- 新增从完整 ONNX 图构建内存常驻 BF16 engine 的初始化流程。同一个 pipeline 实例在其生命周期内为所有帧复用该 engine。
- 新增与 YOLOv5 一致的阶段 banner、进度显示、性能报告，以及独立的端到端/OpenRT 推理 FPS。

### 变更

- 发布记录统一命名为 `ReleaseNote.md`。
- 可复用 helper 统一放在 `src/utils/`。
- 推理封装统一放在 `rpp/inference/`。
- logo 资源移动到 `docs/logo/`。
- RGB 示例图片移动到 `assets/test.jpg`。
- CMake 更新为构建 `rpp_inference` 并使用 V2.0 目录布局。
- 编译流程与模型可用性解耦：`scripts/build.sh` 不再下载或校验 ONNX 文件。
- 运行和 benchmark 流程支持通过 `-m/--model` 指定外部 ONNX 路径，并与 YOLOv5 一致地支持用户自行放置的 `weights/yolov8s.onnx` fallback；发布包本身仍不包含模型。
- demo 二进制参数与 YOLOv5 约定对齐：必须提供 `-m/--model`，输入使用 `-i/--input`，输出使用 `-o/--output`，并支持配置 loop 次数。
- 推理初始化改为直接从用户提供的 ONNX 模型构建；应用不会创建或加载持久化 engine 文件。
- 公开文档统一使用 `ReleaseNote.md`、OpenRT 命名和当前 YOLOv8 模块名称。
- `.gitignore` 更新为覆盖构建产物、生成的运行产物以及用户提供的 ONNX/模型文件。
- 新增已验证的 1280×1024 YOLOv8n/s/m/l/x 模型契约和 RGB 实测性能。

### 修复

- 修复已验证 1280×1024 YOLOv8 输出契约下的 I420 YUV 检测。
- 确保 RGB 和 I420 YUV 工作负载使用完整原生 ONNX 图，不修改用户模型。
- 确保已初始化的 `YoloV8Pipeline` 在 warmup 和正式测量帧之间持续复用内存 engine。

### 移除

- 移除根目录 `build.sh`，统一使用 `scripts/build.sh`。
- 移除旧 `doc/` 布局，公开文档统一放在 `docs/`。
