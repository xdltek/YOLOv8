<a id="english"></a>

# Utility Module

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

This module contains shared helpers used by the demo, inference wrapper, and RPP modules.

## Responsibilities

- Logging wrappers.
- ONNX parser helper.
- RPP binding buffer management.
- Path and CLI argument helpers.
- Runtime error recording helpers used by OpenRT/RPP APIs.

## Files

- `logger.*` and `logging.*`
  Runtime logging helpers.
- `parser_api.h`
  ONNX parser helper declarations.
- `rpp_buffer_manager.h`
  OpenRT/RPP buffer ownership helpers.
- `ErrorRecorder.h`
  Runtime error recording helper.
- `sampleCommon.h`
  Shared sample utilities.
- `utils.hpp`
  CLI argument normalization, path expansion, and small numeric helpers.

## Reuse Notes

Keep utilities model-agnostic. Model-specific labels, preprocessing rules, and postprocessing rules should remain in the YOLOv8 or RPP model modules.

---

<a id="chinese"></a>

# 工具模块

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应语言章节。

本模块包含 demo、推理封装和 RPP 模块共用的辅助代码。

## 模块职责

- 日志封装。
- ONNX parser 辅助。
- RPP binding buffer 管理。
- 路径和命令行参数辅助。
- OpenRT/RPP API 使用的错误记录辅助。

## 文件说明

- `logger.*` 和 `logging.*`
  运行时日志辅助。
- `parser_api.h`
  ONNX parser 辅助声明。
- `rpp_buffer_manager.h`
  OpenRT/RPP buffer 管理辅助。
- `ErrorRecorder.h`
  运行时错误记录辅助。
- `sampleCommon.h`
  共享 sample 工具。
- `utils.hpp`
  CLI 参数归一化、路径展开和小型数值辅助。

## 复用说明

工具模块应保持模型无关。模型标签、前处理规则和后处理规则应保留在 YOLOv8 或 RPP 模型模块中。
