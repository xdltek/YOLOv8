# Release Notes

## RPP-Accelerated YOLOv8 Application V2.0

| Item | Value |
|---|---|
| Release version | V2.0 |
| Target code | Current repository code |
| Validation | `cmake --build build -j8` passed |

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

These measurements describe the local RPP validation environment and require human review before external performance commitments.

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
- Added persistent OpenRT engine caching beside the external model, including model/version/precision cache keys and a pre-deserialization integrity checksum.
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
- Damaged, incomplete, stale, or unchecked engine caches now fall back to rebuilding from ONNX instead of being reused.
- Updated public documentation to use `ReleaseNote.md`, OpenRT naming, and current YOLOv8 module names consistently.
- Updated `.gitignore` coverage for build outputs, generated runtime artifacts, local ONNX/model files, and ignored internal documents.
- Added validated 1280×1024 YOLOv8n/s/m/l/x model contracts and measured RGB performance.

### Removed

- Removed the root `build.sh` entry in favor of `scripts/build.sh`.
- Removed the old `doc/` layout after moving public documentation under `docs/`.
- Removed internal-document generation requirements from the public project flow.

---

## RPP 加速 YOLOv8 应用 V2.0

| 项目 | 内容 |
|---|---|
| 发布版本 | V2.0 |
| 目标代码 | 当前仓库代码 |
| 验证状态 | `cmake --build build -j8` 已通过 |

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

这些数据仅代表本地 RPP 验证环境，用于对外性能承诺前必须进行人工复核。

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
- 新增外部模型同级目录下的 OpenRT engine 持久缓存，缓存键包含模型、版本和精度信息，并在反序列化前执行完整性校验。
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
- engine 缓存损坏、不完整、过期或缺少校验时，自动回退到 ONNX 重建，不再尝试复用。
- 公开文档统一使用 `ReleaseNote.md`、OpenRT 命名和当前 YOLOv8 模块名称。
- `.gitignore` 更新为覆盖构建产物、生成的运行产物、本地 ONNX/模型文件以及被忽略的内部文档。
- 新增已验证的 1280×1024 YOLOv8n/s/m/l/x 模型契约和 RGB 实测性能。

### 移除

- 移除根目录 `build.sh`，统一使用 `scripts/build.sh`。
- 移除旧 `doc/` 布局，公开文档统一放在 `docs/`。
- 从公开项目流程中移除内部文档生成要求。
