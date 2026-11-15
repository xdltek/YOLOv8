<a id="english"></a>

# RPP Postprocess Module

Language switch: click [English](#english) or [中文](#chinese) to jump to the corresponding section.

This module converts YOLOv8 detection output into source-coordinate detection records.

## Responsibilities

- Resolve static `[1,C,N]`, `[C,N]`, or `[N,C]` YOLOv8 detection layouts.
- Transpose channel-major output to row-major form on RPP.
- Fuse FP32-to-BF16 conversion, box/class slicing, score reduction, and NMS in one OpenRT/RppRT engine.
- Restore selected pixel-space `xywh` boxes from letterboxed model coordinates to source coordinates.

## Output Contract

The supported detection head contains four box channels followed directly by class-score channels; YOLOv8 has no separate objectness column:

```text
[x, y, w, h, class_scores...]
```

Validated contracts include:

| Input | Output | Classes |
|---|---|---:|
| `[1,3,640,640]` | `[1,84,8400]` | 80 |
| `[1,3,640,640]` | `[1,605,8400]` | 601 |
| `[1,3,1280,1024]` | `[1,84,26880]` | 80 |

The pipeline derives the class count from the output feature axis. Classification, segmentation, pose, oriented-box, dynamic-shape, and embedded-NMS outputs require separate postprocessing.

## CPU Boundary

Transpose, score reduction, and NMS execute on RPP. The host receives NMS indices plus BF16 boxes and scores, recovers the selected class ID, restores source coordinates, and prepares records for reporting or OpenCV visualization.

## Main API

- `YoloV8Postprocessor::init()` validates the model contract, builds the fused engine, and allocates persistent buffers.
- `YoloV8Postprocessor::run()` executes the fused device stage, transfers result data, and restores selected detections.

---

<a id="chinese"></a>

# RPP 后处理模块

语言切换：点击 [English](#english) 或 [中文](#chinese) 跳转到对应语言章节。

本模块将 YOLOv8 检测输出转换为源图坐标下的检测记录。

## 模块职责

- 解析静态 `[1,C,N]`、`[C,N]` 或 `[N,C]` YOLOv8 检测布局。
- 在 RPP 上将 channel-major 输出转置为 row-major。
- 在一个 OpenRT/RppRT engine 中融合 FP32-to-BF16、框/类别切分、分数归约和 NMS。
- 将像素坐标 `xywh` 检测框从 letterbox 模型坐标恢复到源图坐标。

## 输出契约

支持的检测 head 由 4 个框通道和紧随其后的类别分数通道组成；YOLOv8 不包含独立 objectness 通道：

```text
[x, y, w, h, class_scores...]
```

已验证契约如下：

| 输入 | 输出 | 类别数 |
|---|---|---:|
| `[1,3,640,640]` | `[1,84,8400]` | 80 |
| `[1,3,640,640]` | `[1,605,8400]` | 601 |
| `[1,3,1280,1024]` | `[1,84,26880]` | 80 |

Pipeline 从输出 feature 轴自动推导类别数。分类、分割、姿态、旋转框、动态 shape 和内嵌 NMS 输出需要单独适配后处理。

## CPU 边界

转置、分数归约和 NMS 在 RPP 上执行。Host 接收 NMS 索引以及 BF16 框和分数，恢复选中类别 ID 和源图坐标，并生成用于结果报告或 OpenCV 可视化的记录。

## 主要 API

- `YoloV8Postprocessor::init()` 校验模型契约、构建融合 engine 并分配持久 buffer。
- `YoloV8Postprocessor::run()` 执行融合 device 阶段、传输结果数据并恢复选中的检测结果。
