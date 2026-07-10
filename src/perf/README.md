<a id="english"></a>

# Perf Trace Module

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

This directory contains the optional `rpp_perf` integration used by the YOLOv8 demos. It wraps trace-window creation, scoped host events, and RPP driver-record forwarding behind a small reusable C++ interface.

The module is inactive by default. It becomes active only when the project is configured with `YOLO_ENABLE_RPP_PERF=ON`.

## Files

- `perf_trace_session.h`
  Public helper API, RAII session owner, and `PERF_SCOPE` macros.
- `perf_trace_session.cpp`
  `rpp_perf` implementation guarded by `YOLO_ENABLE_RPP_PERF`.

## Build Configuration

Enable trace support at CMake configure time:

```bash
cmake -S . -B build -DYOLO_ENABLE_RPP_PERF=ON
cmake --build build -j8
```

When the option is off, calls to `PERF_SCOPE` remain valid but become no-ops. If a trace is requested in a non-perf build, the helper prints a rebuild hint for `YOLO_ENABLE_RPP_PERF=ON`.

## CMake Integration

The project builds the perf wrapper as a standalone target:

```cmake
option(YOLO_ENABLE_RPP_PERF "Enable rpp_perf trace integration" OFF)

add_library(perf_trace SHARED
    ${ROOT_DIRECTORY}/src/perf/perf_trace_session.cpp
)

target_include_directories(perf_trace PUBLIC
    ${ROOT_DIRECTORY}/src/perf
)
```

Targets that emit trace scopes link against `perf_trace`. In this project, `yolov8_core`, `rpp_yolo_preprocess`, and `rpp_yolov8_postprocess` already link the perf wrapper through CMake target dependencies.

## Demo Behavior

The RGB and YUV demos automatically create a perf session in perf-enabled builds:

```cpp
PerfTraceSession trace_session("yolov8_rgb_image_demo",
                               kPerfTraceEnabled,
                               "../trace");
trace_session.enableDriverTrace(0);
```

The trace file is written under:

```text
build/trace/
```

The demo prints the generated trace path when the session starts.

## Call Flow

```text
+-------------------------+
| Demo main               |
| create PerfTraceSession |
+-------------------------+
            |
            v
+-------------------------+
| TRACE_START             |
| open rpp_perf window    |
+-------------------------+
            |
            v
+-------------------------+
| enableDriverTrace(0)    |
| attach drv_api records  |
+-------------------------+
            |
            v
+-------------------------+
| PERF_SCOPE_CATE(...)    |
| host stage trace events |
+-------------------------+
            |
            v
+-------------------------+
| TRACE_END_ALL           |
| close JSON trace        |
+-------------------------+
```

## Driver API Records

To make the generated JSON include RPP driver/runtime records, two conditions must both be true:

1. The project is built with:

```bash
-DYOLO_ENABLE_RPP_PERF=ON
```

2. The active session calls:

```cpp
trace_session.enableDriverTrace(0);
```

The current demos pass mode `0` and forward driver records into the same trace window as the host `PERF_SCOPE` events.

If the JSON contains only host demo events and no driver records, check:

- The binary was rebuilt after enabling `YOLO_ENABLE_RPP_PERF`.
- `enableDriverTrace(0)` is called after the session is active.
- The RPP system perf configuration allows driver/device records to be forwarded.
- The process has permission to write into `build/trace`.

## Module Integration

For a demo or library target:

1. Include the header:

```cpp
#include "perf_trace_session.h"
```

2. Define the build-controlled switch:

```cpp
#ifdef YOLO_ENABLE_RPP_PERF
constexpr bool kPerfTraceEnabled = true;
#else
constexpr bool kPerfTraceEnabled = false;
#endif
```

3. Create one session near the start of `main()` after argument parsing:

```cpp
PerfTraceSession trace_session("my_demo_name",
                               kPerfTraceEnabled,
                               "../trace");
trace_session.enableDriverTrace(0);
```

4. Add scoped events around meaningful stages:

```cpp
{
    PERF_SCOPE_CATE("my_demo_preprocess", "preprocess");
    // stage work
}
```

Use stable names and categories so traces from different executable modes keep the same field meaning.

## Scope Naming

Recommended categories:

- `preprocess`
- `inference`
- `postprocess`
- `io`
- `yolov8`

Recommended naming style:

```text
component_stage_action
```

Examples:

```cpp
PERF_SCOPE_CATE("yolov8_pipeline_preprocess_stage", "preprocess");
PERF_SCOPE_CATE("rpp_yolov8_postprocess_run", "postprocess");
```

## Notes

- The trace session owns process-wide trace state. Create one session per process.
- `PerfScope` is intentionally cheap when tracing is disabled.
- Driver trace forwarding is disabled automatically when the session shuts down.
- Demo timing printed to stdout remains independent from `rpp_perf`; stdout timing is always available, while JSON traces require perf-enabled builds.

---

<a id="chinese"></a>

# Perf Trace 模块

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应语言章节。

本目录包含 YOLOv8 demo 使用的可选 `rpp_perf` 集成。它通过一个小型 C++ 接口封装 trace window 创建、host 侧 scope 事件和 RPP driver record 转发。

默认情况下该模块不启用。只有在 CMake 配置 `YOLO_ENABLE_RPP_PERF=ON` 时才会生效。

## 文件

- `perf_trace_session.h`
  公共辅助 API、RAII session owner 和 `PERF_SCOPE` 宏。
- `perf_trace_session.cpp`
  由 `YOLO_ENABLE_RPP_PERF` 保护的 `rpp_perf` 实现。

## 构建配置

在 CMake 配置阶段启用 trace 支持：

```bash
cmake -S . -B build -DYOLO_ENABLE_RPP_PERF=ON
cmake --build build -j8
```

未开启该选项时，`PERF_SCOPE` 调用仍可编译，但会变成 no-op。如果在非 perf 构建中请求 trace，辅助模块会打印 `YOLO_ENABLE_RPP_PERF=ON` 的重新构建提示。

## CMake 集成

工程将 perf wrapper 构建为独立 target：

```cmake
option(YOLO_ENABLE_RPP_PERF "Enable rpp_perf trace integration" OFF)

add_library(perf_trace SHARED
    ${ROOT_DIRECTORY}/src/perf/perf_trace_session.cpp
)

target_include_directories(perf_trace PUBLIC
    ${ROOT_DIRECTORY}/src/perf
)
```

会发出 trace scope 的 target 需要链接 `perf_trace`。本工程中，`yolov8_core`、`rpp_yolo_preprocess` 和 `rpp_yolov8_postprocess` 已经通过 CMake target 依赖链接该 wrapper。

## Demo 行为

RGB 和 YUV demo 会在 perf-enabled 构建中自动创建 perf session：

```cpp
PerfTraceSession trace_session("yolov8_rgb_image_demo",
                               kPerfTraceEnabled,
                               "../trace");
trace_session.enableDriverTrace(0);
```

trace 文件写入：

```text
build/trace/
```

session 启动时，demo 会打印生成的 trace 路径。

## 调用流程

```text
+-------------------------+
| Demo main               |
| 创建 PerfTraceSession   |
+-------------------------+
            |
            v
+-------------------------+
| TRACE_START             |
| 打开 rpp_perf window    |
+-------------------------+
            |
            v
+-------------------------+
| enableDriverTrace(0)    |
| 关联 drv_api records    |
+-------------------------+
            |
            v
+-------------------------+
| PERF_SCOPE_CATE(...)    |
| host 阶段 trace 事件    |
+-------------------------+
            |
            v
+-------------------------+
| TRACE_END_ALL           |
| 关闭 JSON trace         |
+-------------------------+
```

## Driver API Records

要让生成的 JSON 包含 RPP driver/runtime records，需要同时满足两个条件：

1. 工程使用如下选项构建：

```bash
-DYOLO_ENABLE_RPP_PERF=ON
```

2. 当前 session 调用：

```cpp
trace_session.enableDriverTrace(0);
```

当前 demo 传入 mode `0`，并把 driver records 转发到与 host `PERF_SCOPE` 事件相同的 trace window 中。

如果 JSON 中只有 host demo 事件而没有 driver records，请检查：

- 二进制是否在启用 `YOLO_ENABLE_RPP_PERF` 后重新构建。
- `enableDriverTrace(0)` 是否在 session 激活后调用。
- RPP 系统 perf 配置是否允许转发 driver/device records。
- 当前进程是否有权限写入 `build/trace`。

## 模块集成

对于 demo 或 library target：

1. 包含头文件：

```cpp
#include "perf_trace_session.h"
```

2. 定义构建控制开关：

```cpp
#ifdef YOLO_ENABLE_RPP_PERF
constexpr bool kPerfTraceEnabled = true;
#else
constexpr bool kPerfTraceEnabled = false;
#endif
```

3. 在 `main()` 中参数解析之后创建一个 session：

```cpp
PerfTraceSession trace_session("my_demo_name",
                               kPerfTraceEnabled,
                               "../trace");
trace_session.enableDriverTrace(0);
```

4. 在有意义的阶段周围添加 scope 事件：

```cpp
{
    PERF_SCOPE_CATE("my_demo_preprocess", "preprocess");
    // stage work
}
```

使用稳定的名称和类别，保证不同可执行模式生成的 trace 字段含义一致。

## Scope 命名

推荐类别：

- `preprocess`
- `inference`
- `postprocess`
- `io`
- `yolov8`

推荐命名风格：

```text
component_stage_action
```

示例：

```cpp
PERF_SCOPE_CATE("yolov8_pipeline_preprocess_stage", "preprocess");
PERF_SCOPE_CATE("rpp_yolov8_postprocess_run", "postprocess");
```

## 说明

- trace session 持有进程级 trace 状态。每个进程创建一个 session。
- tracing 关闭时，`PerfScope` 的开销很低。
- session 关闭时会自动禁用 driver trace 转发。
- demo 打印到 stdout 的计时与 `rpp_perf` 相互独立；stdout 计时始终可用，JSON trace 需要 perf-enabled 构建。
