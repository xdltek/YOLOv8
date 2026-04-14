# YOLOv8 Inference Demo

![XDL Logo](doc/logo/logo_color_horizontal.png)

This project is a C++ YOLOv8 object detection demo that uses RppRT for inference and OpenCV for image loading, preprocessing, and visualization.

The current implementation is centered on [`yolo/main.cpp`](/home/azurengine/xdl_github/YOLOv8/yolo/main.cpp), [`yolo/yolo.cpp`](/home/azurengine/xdl_github/YOLOv8/yolo/yolo.cpp), and [`CMakeLists.txt`](/home/azurengine/xdl_github/YOLOv8/CMakeLists.txt).

## Overview

The demo loads a YOLOv8 ONNX model, preprocesses the input image with a letterbox-style resize, runs RPP inference, decodes the planar output tensor, applies NMS, and saves the rendered result image.

Current implementation highlights:

- RPP-only inference flow
- YOLOv8 output decoding for `[C, N]` and `[1, C, N]` tensor layouts
- Built-in COCO-80 class labels
- Timing output for single-run and looped inference

## Project Structure

```text
YOLOv8/
├── 3rd_party/
│   └── argparse/argparse.hpp
├── common/
│   ├── logger.cpp
│   ├── logger.h
│   ├── logging.cpp
│   ├── logging.h
│   ├── parser_api.h
│   ├── rpp_buffer_manager.h
│   ├── sampleCommon.h
│   ├── ErrorRecorder.h
│   └── utils.hpp
├── doc/
│   ├── images/test.jpg
│   └── logo/logo_color_horizontal.png
├── yolo/
│   ├── main.cpp
│   ├── yolo.cpp
│   └── yolo.h
├── CMakeLists.txt
├── LICENSE
└── README.md
```

## How It Works

1. Parse CLI arguments in `main.cpp`
2. Load the image with OpenCV
3. Build the RppRT engine with `Yolo::init_engine()`
4. Convert the image into a float NCHW blob with letterbox preprocessing
5. Run warmup plus timed inference through `Yolo::infer()`
6. Decode detections and apply NMS
7. Draw boxes and save `output.jpg`

## Dependencies

Required:

- CMake 3.10+
- C++17 compiler
- OpenCV with `core`, `imgproc`, and `imgcodecs`
- RppRT installed under `/usr/local/rpp` as expected by `CMakeLists.txt`

Example on Debian/Ubuntu:

```bash
sudo apt-get install -y cmake build-essential libopencv-dev
```

## Build

```bash
cd YOLOv8
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

The executable name is:

```bash
./yolov8_demo
```

## Run

```bash
./yolov8_demo \
  --onnx /path/to/yolov8.onnx \
  --image /path/to/image.jpg \
  --loop 1
```

## CLI Arguments

| Option | Description |
| --- | --- |
| `-o`, `--onnx` | Required path to the YOLOv8 ONNX model |
| `-i`, `--image` | Input image path. The default sample is `doc/images/test.jpg` |
| `--loop` | Number of timed inference iterations |
| `-v`, `--verbose` | Enable verbose RPP logging |

## Output

The demo writes:

- `output.jpg`

It also prints inference timing and FPS information to the console.

## Notes

- The preprocessing path assumes a 640x640 model input
- Output decoding expects a planar YOLOv8 tensor layout with box channels followed by class scores
- Class labels are fixed to COCO-80 in the source
- If `--image` is omitted, the demo falls back to the bundled sample image `doc/images/test.jpg`

## Troubleshooting

| Problem | Suggestion |
| --- | --- |
| Model file not found | Use an absolute path with `--onnx` |
| Image file not found | Pass `--image` explicitly |
| RPP link or runtime errors | Verify RppRT is installed under `/usr/local/rpp` |
| Incorrect boxes | Check that the ONNX export matches the YOLOv8 planar output assumptions |

## License

See [`LICENSE`](/home/azurengine/xdl_github/YOLOv8/LICENSE).
