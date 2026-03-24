# YOLOv8m Demo Guide

## Overview

The YOLOv8m demo is an object detection application that uses the YOLO (You Only Look Once) algorithm to detect objects in images in real-time. This demo processes images and outputs annotated images with bounding boxes, confidence scores, and class labels for detected objects.

### Key Features
- **Model**: YOLOv8m (medium variant)
- **Input**: Images resized to 640×640 pixels
- **Output**: Annotated images with detected objects marked by bounding boxes and labels
- **Precision**: Supports both float32 (BF16) and INT8 quantization
- **Framework**: Uses OpenRT inference engine


## Code Structure

### Main Components

1. **yolo_main.cpp**: Entry point, handles command-line arguments and orchestrates the inference pipeline
2. **yolov8.h**: Header file defining the `Yolov8s` class and data structures
3. **yolov8.cpp**: Implementation of preprocessing, post-processing, NMS, and drawing functions

### Key Classes and Functions

#### Yolov8s Class
- Inherits from `SampleModel` (base inference class)
- Implements:
  - `PreProcess()`: Image preprocessing (resize, normalization, format conversion)
  - `PostProcess()`: Post-processing (NMS, drawing bounding boxes)
  - `NMS()`: Non-Maximum Suppression for filtering overlapping detections
  - `Draw()`: Draws bounding boxes and labels on the image
  - `IOUCalculate()`: Calculates Intersection over Union for NMS

#### Data Structures
- `DetectRes`: Contains detection results (bounding box coordinates, class, confidence)
- `ClassRes`: Contains class ID and probability
- `DetectMask`: Contains mask coordinates (for pose estimation, if applicable)

### Processing Pipeline

1. **Preprocessing** (`PreProcess`):
   - Load image from file
   - Resize to 640×640 while maintaining aspect ratio
   - Normalize pixel values to [0, 1]
   - Convert BGR to RGB
   - Prepare input buffer for inference

2. **Inference**:
   - Execute model inference using OpenRT engine
   - Model outputs: 8400×84 tensor (8400 detections, 84 features per detection)

3. **Post-processing** (`PostProcess`):
   - Parse model output
   - Apply confidence threshold (0.25)
   - Scale coordinates back to original image size
   - Apply Non-Maximum Suppression (NMS) with IoU threshold 0.45
   - Draw bounding boxes and labels

## Configuration

### Model Configuration File

The `yolov8m_config.json` file contains inference engine settings:

```json
{
  "yolov8m_config.json": [
    {
      "fusion_config": "",
      "dyn_freq_config": "auto",
      "use_dyn_freq": false,
      "use_weight_int4": false,
      "sram_default_size_efficiency": 0.9,
      "sram_stack_size": 1048576,
      "weight_buffer": 1048576,
      "bfloat16 inputs": [],
      "bfloat16 outputs": [],
      "infer_flag": ["YOLOV8M_CONV_SWISH_FUSE"]
    }
  ]
}
```

### Model Parameters

- **Input Size**: 640×640 pixels
- **Output Size**: 8400×84 (8400 detections, 84 features)
- **Confidence Threshold**: 0.25
- **NMS IoU Threshold**: 0.45
- **Precision**: BF16 (float demo) or INT8 (quantized demo)

## Troubleshooting

### Common Issues

1. **Model file not found**
   ```
   Cannot found onnx model file, path: yolov8m.onnx
   ```
   **Solution**: Ensure the ONNX model file exists in the current directory or provide full path with `-o` option

2. **Image file not found**
   ```
   Cannot found image file, path: ./images/test_0.jpg
   ```
   **Solution**: 
   - Check that the image file exists
   - Use `-i` option to specify a valid image path
   - Ensure images directory exists with test images

3. **Build failed**
   **Solution**: 
   - Verify all dependencies are installed
   - Check that `SAMPLE_ROOT` environment variable is set
   - Ensure CMake version is 3.10 or higher

4. **Inference failed**
   ```
   Infer Yolov8m failed
   ```
   **Solution**:
   - Check GPU/driver availability
   - Verify model file is valid ONNX format
   - Enable verbose logging with `--verbose` for detailed error messages

### Performance Tips

- Use `--loop` option to measure average inference time
- For batch processing, modify the code to process multiple images in sequence
- INT8 quantized model (`yolov8m.int8.onnx`) provides faster inference with slightly reduced accuracy

## Advanced Usage

### Processing Multiple Images

The demo can process multiple images from the `./images/` directory automatically. If no `-i` option is specified, it will cycle through images in the directory.

### Custom Class Labels

To use custom class labels:
1. Create or modify `yolov8.names` file (one class name per line)
2. Ensure the class count matches your model's output classes

## Writing Custom YOLO Inference Code

This section provides a comprehensive guide for customers to write their own YOLO inference code from scratch, covering all four essential steps: **Preprocessing**, **Build**, **Inference**, and **Post-processing**.

### Overview

To implement YOLO inference, you need to:
1. **Preprocess** images: resize, normalize, and format for the model
2. **Build** the inference engine: parse ONNX model and create optimized engine
3. **Run Inference**: execute the model on preprocessed data
4. **Post-process** results: parse outputs, apply NMS, and draw detections

### Step 1: Create Your Custom YOLO Class

First, create a header file (`my_yolo.h`) that inherits from `SampleModel`:

```cpp
#pragma once
#include "infer/infer_sample_base.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <map>

// Detection result structure
struct Detection {
    float x, y, w, h;      // Bounding box (center x, center y, width, height)
    float confidence;      // Detection confidence score
    int class_id;          // Detected class ID
};

class MyYolo : public SampleModel {
public:
    explicit MyYolo(const samplesCommon::ModelSampleParams& params);
    
    // Override required virtual functions
    bool PreProcess(const samplesCommon::RppBufferManager& buffers) override;
    bool PostProcess(const samplesCommon::RppBufferManager& buffers) override;

private:
    // Model-specific parameters
    static constexpr int INPUT_WIDTH = 640;
    static constexpr int INPUT_HEIGHT = 640;
    static constexpr int OUTPUT_HEIGHT = 84;
    static constexpr int OUTPUT_WIDTH = 8400;
    static constexpr float CONF_THRESHOLD = 0.25f;
    static constexpr float NMS_THRESHOLD = 0.45f;
    
    // Helper functions
    std::vector<Detection> parseOutput(const float* output, int img_width, int img_height);
    std::vector<Detection> applyNMS(std::vector<Detection>& detections);
    float calculateIoU(const Detection& a, const Detection& b);
    void drawDetections(cv::Mat& img, const std::vector<Detection>& detections);
    std::map<int, std::string> loadClassLabels(const std::string& filename);
    
    std::map<int, std::string> class_labels_;
    int infer_count_ = 0;
};
```

### Step 2: Implement Preprocessing

The `PreProcess` function prepares input images for the model. Here's the complete implementation:

```cpp
#include "my_yolo.h"
#include "logger.h"

MyYolo::MyYolo(const samplesCommon::ModelSampleParams& params) 
    : SampleModel(params) {
    // Load class labels
    class_labels_ = loadClassLabels(mParams.referenceFileName);
}

bool MyYolo::PreProcess(const samplesCommon::RppBufferManager& buffers) {
    // Step 1: Load image from file
    cv::Mat image = cv::imread(mParams.inferImageFileName, cv::IMREAD_COLOR);
    if (image.empty()) {
        sample::LOG_ERROR() << "Could not open image: " 
                           << mParams.inferImageFileName << std::endl;
        return false;
    }
    
    int orig_height = image.rows;
    int orig_width = image.cols;
    
    // Step 2: Calculate resize ratio (maintain aspect ratio)
    float ratio = std::min(
        float(INPUT_WIDTH) / float(orig_width),
        float(INPUT_HEIGHT) / float(orig_height)
    );
    
    // Step 3: Resize image while maintaining aspect ratio
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(), ratio, ratio);
    
    // Step 4: Create padded image (640x640) with black borders
    cv::Mat padded = cv::Mat::zeros(cv::Size(INPUT_WIDTH, INPUT_HEIGHT), CV_8UC3);
    resized.copyTo(padded(cv::Rect(0, 0, resized.cols, resized.rows)));
    
    // Step 5: Normalize pixel values to [0, 1] range
    cv::Mat normalized;
    padded.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
    
    // Step 6: Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(normalized, rgb, cv::COLOR_BGR2RGB);
    
    // Step 7: Split channels and reorder to CHW format (Channel-Height-Width)
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    
    // Step 8: Prepare input buffer (CHW format: R, G, B)
    int buffer_size = samplesCommon::volume(mInputDims) * 
                      samplesCommon::getElementSize(mInputDataType);
    void* input_buffer = buffers.getHostBuffer(mInOut["input"]);
    
    // Copy channels in R, G, B order
    int channel_size = INPUT_WIDTH * INPUT_HEIGHT * sizeof(float);
    memcpy((char*)input_buffer + 0 * channel_size, channels[0].data, channel_size);
    memcpy((char*)input_buffer + 1 * channel_size, channels[1].data, channel_size);
    memcpy((char*)input_buffer + 2 * channel_size, channels[2].data, channel_size);
    
    return true;
}
```

**Key Points:**
- Images are resized to 640×640 while maintaining aspect ratio
- Padding with black borders ensures fixed input size
- Pixel values are normalized from [0, 255] to [0, 1]
- Color format conversion: BGR → RGB
- Data layout: CHW (Channel-Height-Width) format required by the model

### Step 3: Build the Inference Engine

The `build()` function (inherited from `SampleModel`) handles engine creation. You only need to configure parameters:

```cpp
// In your main function or initialization code
samplesCommon::ModelSampleParams params;

// Required parameters
params.onnxFileName = "yolov8m.onnx";           // Path to ONNX model
params.referenceFileName = "yolov8.names";       // Path to class labels file
params.inferImageFileName = "input_image.jpg";   // Input image path
params.model_config = "yolov8m_config.json";    // Model configuration file

// Optional parameters
params.batchSize = 1;                            // Batch size (usually 1 for YOLO)
params.loops = 1;                                // Number of inference loops
params.int8 = false;                             // Use INT8 quantization
params.bf16 = true;                              // Use BF16 precision
params.check_inf_nan = false;                    // Check for invalid values

// Image directory (if processing multiple images)
params.dataDirs.push_back("./images/");
params.image_count = 1;                          // Number of images to process

// Create your YOLO instance
MyYolo yolo(params);

// Build the inference engine
if (!yolo.build()) {
    std::cerr << "Failed to build inference engine!" << std::endl;
    return -1;
}
```

**What happens during `build()`:**
1. Creates a builder and network definition
2. Parses the ONNX model file
3. Applies model configuration (optimizations, precision settings)
4. Builds an optimized inference engine
5. Extracts input/output tensor names and dimensions
6. Loads image file names from specified directories

### Step 4: Run Inference

The `infer()` function (also inherited) handles the inference loop:

```cpp
// Run inference
if (!yolo.infer()) {
    std::cerr << "Inference failed!" << std::endl;
    return -1;
}

// Cleanup
yolo.teardown();
```

**What happens during `infer()`:**
1. Creates execution context and buffer manager
2. For each image:
   - Calls `PreProcess()` to prepare input
   - Copies input data to device (GPU)
   - Executes inference (runs model forward pass)
   - Copies output data back to host
   - Calls `PostProcess()` to handle results
3. Reports inference timing and throughput

### Step 5: Implement Post-processing

The `PostProcess` function parses model outputs and applies NMS:

```cpp
bool MyYolo::PostProcess(const samplesCommon::RppBufferManager& buffers) {
    // Step 1: Get output data from buffer
    const float* output_ptr = static_cast<const float*>(
        buffers.getHostBuffer(mInOut.at("output"))
    );
    
    int output_size = mOutputDims.d[0] * mOutputDims.d[1];
    std::vector<float> output(output_ptr, output_ptr + output_size);
    
    // Step 2: Load original image for coordinate scaling
    cv::Mat img = cv::imread(mParams.inferImageFileName);
    if (img.empty()) {
        return false;
    }
    
    // Step 3: Parse model output to get detections
    std::vector<Detection> detections = parseOutput(
        output.data(), img.cols, img.rows
    );
    
    // Step 4: Apply Non-Maximum Suppression
    std::vector<Detection> filtered = applyNMS(detections);
    
    // Step 5: Draw detections on image
    drawDetections(img, filtered);
    
    // Step 6: Save output image
    std::string output_name = "output_" + std::to_string(infer_count_++) + ".jpg";
    cv::imwrite(output_name, img);
    
    sample::user_visible_stream_log("Output saved to: ", output_name);
    return true;
}

std::vector<Detection> MyYolo::parseOutput(
    const float* output, int img_width, int img_height) {
    
    std::vector<Detection> detections;
    
    // Calculate scaling ratios (model input vs original image)
    float ratio_w = float(img_width) / float(INPUT_WIDTH);
    float ratio_h = float(img_height) / float(INPUT_HEIGHT);
    float ratio = std::max(ratio_w, ratio_h);
    
    // Model output shape: [8400, 84]
    // Each row: [cx, cy, w, h, class_scores...] (80 classes)
    cv::Mat output_mat(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_32F, (float*)output);
    cv::Mat output_transposed = output_mat.t();  // [8400, 84]
    
    for (int i = 0; i < OUTPUT_WIDTH; i++) {
        // Extract bounding box coordinates
        float cx = output_transposed.at<float>(i, 0);
        float cy = output_transposed.at<float>(i, 1);
        float w = output_transposed.at<float>(i, 2);
        float h = output_transposed.at<float>(i, 3);
        
        // Extract class scores (columns 4-83)
        cv::Mat class_scores = output_transposed.row(i).colRange(4, OUTPUT_HEIGHT);
        cv::Point class_id_point;
        double confidence;
        cv::minMaxLoc(class_scores, nullptr, &confidence, nullptr, &class_id_point);
        
        // Filter by confidence threshold
        if (confidence > CONF_THRESHOLD) {
            Detection det;
            det.x = cx * ratio;           // Scale to original image
            det.y = cy * ratio;
            det.w = w * ratio;
            det.h = h * ratio;
            det.confidence = confidence;
            det.class_id = class_id_point.x;
            detections.push_back(det);
        }
    }
    
    return detections;
}

std::vector<Detection> MyYolo::applyNMS(std::vector<Detection>& detections) {
    // Sort by confidence (descending)
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });
    
    // Apply NMS: remove overlapping detections of the same class
    for (size_t i = 0; i < detections.size(); i++) {
        if (detections[i].confidence == 0) continue;
        
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (detections[j].confidence == 0) continue;
            if (detections[i].class_id != detections[j].class_id) continue;
            
            float iou = calculateIoU(detections[i], detections[j]);
            if (iou > NMS_THRESHOLD) {
                detections[j].confidence = 0;  // Suppress this detection
            }
        }
    }
    
    // Remove suppressed detections
    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
            [](const Detection& d) { return d.confidence == 0; }),
        detections.end()
    );
    
    return detections;
}

float MyYolo::calculateIoU(const Detection& a, const Detection& b) {
    // Calculate intersection area
    float x1 = std::max(a.x - a.w/2, b.x - b.w/2);
    float y1 = std::max(a.y - a.h/2, b.y - b.h/2);
    float x2 = std::min(a.x + a.w/2, b.x + b.w/2);
    float y2 = std::min(a.y + a.h/2, b.y + b.h/2);
    
    if (x2 < x1 || y2 < y1) return 0.0f;
    
    float intersection = (x2 - x1) * (y2 - y1);
    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    float union_area = area_a + area_b - intersection;
    
    if (union_area == 0) return 0.0f;
    return intersection / union_area;
}

void MyYolo::drawDetections(cv::Mat& img, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        // Calculate bounding box coordinates
        int x1 = static_cast<int>(det.x - det.w / 2);
        int y1 = static_cast<int>(det.y - det.h / 2);
        int x2 = static_cast<int>(det.x + det.w / 2);
        int y2 = static_cast<int>(det.y + det.h / 2);
        
        // Clamp to image boundaries
        x1 = std::max(0, std::min(x1, img.cols - 1));
        y1 = std::max(0, std::min(y1, img.rows - 1));
        x2 = std::max(0, std::min(x2, img.cols - 1));
        y2 = std::max(0, std::min(y2, img.rows - 1));
        
        // Draw bounding box
        cv::Rect box(x1, y1, x2 - x1, y2 - y1);
        cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);
        
        // Draw label background
        std::string label = class_labels_[det.class_id] + 
                           " " + std::to_string(det.confidence).substr(0, 4);
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                            0.6, 2, &baseline);
        cv::rectangle(img, cv::Point(x1, y1 - text_size.height - 10),
                     cv::Point(x1 + text_size.width, y1),
                     cv::Scalar(0, 255, 0), -1);
        
        // Draw label text
        cv::putText(img, label, cv::Point(x1, y1 - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
}

std::map<int, std::string> MyYolo::loadClassLabels(const std::string& filename) {
    std::map<int, std::string> labels;
    std::ifstream file(filename);
    if (!file.is_open()) {
        sample::LOG_ERROR() << "Cannot open class labels file: " << filename << std::endl;
        return labels;
    }
    
    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        labels[index++] = line;
    }
    file.close();
    return labels;
}
```

### Complete Example: Main Function

Here's a complete example showing how to use your custom YOLO class:

```cpp
#include "my_yolo.h"
#include "logger.h"
#include <iostream>

int main(int argc, char* argv[]) {
    // Configure parameters
    samplesCommon::ModelSampleParams params;
    params.onnxFileName = "yolov8m.onnx";
    params.referenceFileName = "yolov8.names";
    params.inferImageFileName = "test_image.jpg";
    params.model_config = "yolov8m_config.json";
    params.loops = 1;
    params.int8 = false;
    params.bf16 = true;
    params.batchSize = 1;
    
    // Validate files exist
    if (!std::filesystem::exists(params.onnxFileName)) {
        std::cerr << "Model file not found: " << params.onnxFileName << std::endl;
        return -1;
    }
    
    // Create and initialize YOLO instance
    MyYolo yolo(params);
    
    // Step 1: Build inference engine
    sample::LOG_INFO() << "Building inference engine..." << std::endl;
    if (!yolo.build()) {
        sample::LOG_ERROR() << "Failed to build engine!" << std::endl;
        return -1;
    }
    
    // Step 2: Run inference
    sample::LOG_INFO() << "Running inference..." << std::endl;
    if (!yolo.infer()) {
        sample::LOG_ERROR() << "Inference failed!" << std::endl;
        return -1;
    }
    
    // Step 3: Cleanup
    yolo.teardown();
    sample::LOG_INFO() << "Inference completed successfully!" << std::endl;
    
    return 0;
}
```

### Key Implementation Details

#### Preprocessing Checklist:
- ✅ Load image using OpenCV
- ✅ Resize maintaining aspect ratio
- ✅ Pad to fixed size (640×640)
- ✅ Normalize to [0, 1] range
- ✅ Convert BGR to RGB
- ✅ Arrange data in CHW format
- ✅ Copy to input buffer

#### Build Checklist:
- ✅ Set ONNX model path
- ✅ Configure precision (BF16/INT8)
- ✅ Set model configuration file
- ✅ Specify input/output names (auto-detected)
- ✅ Call `build()` to create engine

#### Inference Checklist:
- ✅ Create buffer manager
- ✅ Call `PreProcess()` for each image
- ✅ Copy input to device
- ✅ Execute inference context
- ✅ Copy output to host
- ✅ Call `PostProcess()` for results

#### Post-processing Checklist:
- ✅ Extract output tensor data
- ✅ Parse bounding boxes and scores
- ✅ Apply confidence threshold (0.25)
- ✅ Scale coordinates to original image size
- ✅ Apply NMS to remove duplicates
- ✅ Draw bounding boxes and labels
- ✅ Save annotated image

### Common Customization Points

1. **Change Input Size**: Modify `INPUT_WIDTH` and `INPUT_HEIGHT` constants
2. **Adjust Thresholds**: Change `CONF_THRESHOLD` and `NMS_THRESHOLD` values
3. **Custom Drawing**: Modify `drawDetections()` for different visualization styles
4. **Batch Processing**: Set `params.batchSize > 1` and modify preprocessing accordingly
5. **Multiple Images**: Set `params.image_count` and provide image directory

### Troubleshooting Tips

- **Wrong input format**: Ensure CHW layout and RGB color order
- **Incorrect coordinates**: Verify scaling ratios match preprocessing
- **No detections**: Check confidence threshold isn't too high
- **Too many detections**: Lower confidence threshold or increase NMS threshold
- **Memory issues**: Reduce batch size or input resolution

## Related Demos

- **YOLOv8m INT8 Demo**: Located at `build/demos/cpp_demos/yolo_quant/`
- **YOLOv5 Demo**: Alternative YOLO implementation at `build/demos/cpp_demos/yolov5/`

## Additional Resources

- Main README: `$SAMPLE_ROOT/demos/cpp_demos/README.md`
- Model Zoo: `$SAMPLE_ROOT/onnx_model_zoo/`
- Configuration files: `$SAMPLE_ROOT/demos/cpp_demos/common/openrt_configs/`

## Summary

The YOLOv8m demo provides a complete object detection pipeline:

1. **Build**: `cd $SAMPLE_ROOT && mkdir build && cd build && cmake .. && make`
2. **Run**: `cd build/demos/cpp_demos/yolo && ./yolov8m_demo`
3. **Customize**: Use `-i` for custom images, `-o` for custom models, `--verbose` for debugging
4. **Verify**: Check `output_0.jpg` for detection results

For more information, refer to the main project documentation and README files.

