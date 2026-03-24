/**
 * @file yolov8.cpp
 * @brief RPP pipeline implementation for YOLO demo. See README.md for full workflow.
 */
#include "yolov8.h"
#include "logger.h"

/**
 * @brief Construct YOLO demo model wrapper with default input/output geometry.
 * @param params Runtime parameters for model build and inference.
 */
Yolov8s::Yolov8s(const samplesCommon::ModelSampleParams& params): SampleModel(params){
    input_width_ = 640;
    input_height_ = 640;
    output_height_ = 84;
    output_width_ = 8400;
    infer_count_ = 0;
}

/**
 * @brief Decode model output, run NMS, and draw detections.
 * @param buffers Host/device buffer manager containing model outputs.
 * @return true on success.
 */
bool Yolov8s::PostProcess(const samplesCommon::RppBufferManager &buffers) {
    // Sync postprocess tensor shape with actual engine output dims.
    output_height_ = static_cast<int>(mOutputDims.d[0]);
    output_width_ = static_cast<int>(mOutputDims.d[1]);

    const float* probPtr = static_cast<const float*>(buffers.getHostBuffer(mInOut.at("output")));
    std::vector<float> output(probPtr, probPtr + mOutputDims.d[0] * mOutputDims.d[1]);
    // Optional NaN/Inf guard for debug validation mode.
    if(mParams.check_inf_nan)
    {
        for(auto it : output)
        {
            if(std::isinf(it) || std::isnan(it))
            {
                sample::user_visible_stream_log("Yolov8s result is error, please check!!!!");
                exit(0);
            }
        }
    }
    // Load original image for visualization in original resolution.
    cv::Mat src_img = cv::imread( mParams.inferImageFileName);
    std::vector<DetectRes> result = NMS(output, src_img);

    Draw(src_img, result);
    return true;
}

/**
 * @brief Prepare one input image and write NCHW float tensor to input buffer.
 * @param buffers Host/device buffer manager containing input binding memory.
 * @return true on success, false if no valid image is found.
 */
bool Yolov8s::PreProcess(const samplesCommon::RppBufferManager& buffers)
{
    if (mParams.inferImageFileName.empty()) {
        if (image_file_names_.empty()) {
            sample::LOG_ERROR() << "Cannot found any image file" << std::endl;
            return false;
        }

        int index = image_index % image_file_names_.size();
        image_index++;
        mParams.inferImageFileName = image_file_names_.at(index);
    }

    // sample::LOG_CONSOLE() << "input image: " << mParams.inferImageFileName << std::endl;
    // From image file.
    // Compute byte size for input tensor copy.
    int outSize = samplesCommon::volume(mInputDims) * samplesCommon::getElementSize(mInputDataType);

    cv::Mat image = cv::imread(mParams.inferImageFileName, cv::IMREAD_COLOR);
    if (image.empty()) // Check for invalid input
    {
        sample::LOG_ERROR() << "Could not open or find the image file: " << mParams.inferImageFileName << std::endl;
        return false;
    }
    // Resize with aspect ratio and pad to model input size.
    int image_height = image.rows;
    int image_width = image.cols;
    float ratio = float(input_width_) / float(image_width) < float(input_height_) / float(image_height) ?
                  float(input_width_) / float(image_width) : float(input_height_) / float(image_height);
    cv::Mat flt_img = cv::Mat::zeros(cv::Size(input_width_, input_height_), CV_8UC3);
    cv::resize(image, image, cv::Size(), ratio, ratio);
    image.copyTo(flt_img(cv::Rect(0, 0, image.cols, image.rows)));
    image = flt_img;

    // Convert BGR HWC image to normalized RGB NCHW tensor layout.
    cv::Mat image_float;
    image.convertTo(image_float, CV_32FC3, 1.0 / 255);
    cv::cvtColor(image_float, image_float, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> mv(3);
    cv::split(image_float, mv);
    cv::Mat out;
    out.push_back(mv[2]);
    out.push_back(mv[1]);
    out.push_back(mv[0]);
    memcpy(buffers.getHostBuffer(mInOut["input"]), out.data, outSize);
    return true;
}

/**
 * @brief Compute CIoU-like score between two center-format boxes.
 * @param det_a First detection.
 * @param det_b Second detection.
 * @return Overlap score used by NMS suppression.
 */
float Yolov8s::IOUCalculate(const DetectRes &det_a, const DetectRes &det_b)
{
    cv::Point2f center_a(det_a.x, det_a.y);
    cv::Point2f center_b(det_b.x, det_b.y);
    cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                        std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
    cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
                           std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
    float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
    float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
    float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}

/**
 * @brief Parse raw network output and perform class-wise NMS.
 * @param output Flat output tensor data.
 * @param img Input image for coordinate scaling.
 * @return Filtered detections.
 */
std::vector<DetectRes> Yolov8s::NMS(std::vector<float> output, cv::Mat &img)
{
    std::vector<DetectRes> result;
    float ratioW = float(img.cols) / float(input_width_);
    float ratioH = float(img.rows) / float(input_height_);
    float ratio = ratioW > ratioH ? ratioW : ratioH;
    ratioW = ratioH = ratio;
    size_t outH = output_height_;
    size_t outW = output_width_;
    // Transpose output to candidate-major view for easier decoding.
    cv::Mat dout(outH, outW, CV_32F, (float *)output.data());
    cv::Mat det_output = dout.t();
    for (int i = 0; i < det_output.rows; i++) {
        cv::Mat classes_scores = det_output.row(i).colRange(4, outH);
        cv::Point classIdPoint;
        double score;
        cv::minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
        if (score > 0.25) {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            // int x = static_cast<int>((cx - 0.5 * ow) * ratioW);
            // int y = static_cast<int>((cy - 0.5 * oh) * ratioH);
            int x = static_cast<int>(cx * ratioW);
            int y = static_cast<int>(cy * ratioH);
            int width = static_cast<int>(ow * ratioW);
            int height = static_cast<int>(oh * ratioH);
            DetectRes box;
            box.x = x;
            box.y = y;
            box.w = width;
            box.h = height;
            box.prob = score;
            box.classes = classIdPoint.x;
            result.push_back(box);
        }
    }

    // Sort by confidence so higher-score boxes are kept first.
    sort(result.begin(), result.end(), [=](const DetectRes &left, const DetectRes &right)
    { return left.prob > right.prob; });

    for (int i = 0; i < (int)result.size(); i++)
        for (int j = i + 1; j < (int)result.size(); j++)
        {
            if (result[i].classes == result[j].classes)
            {
                float iou = IOUCalculate(result[i], result[j]);
                if (iou > 0.45)
                    result[j].prob = 0;
            }
        }

    result.erase(std::remove_if(result.begin(), result.end(), [](const DetectRes &det)
                 { return det.prob == 0; }),
                 result.end());
    return result;
}

/**
 * @brief Draw bounding boxes and confidence labels on output image.
 * @param img Image to draw on.
 * @param result Detection results after NMS.
 */
void Yolov8s::Draw(cv::Mat &img, std::vector<DetectRes> &result)
{
    for (const auto &tracker_box : result)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.1f%%", tracker_box.prob * 100.0);

        cv::Rect rst(tracker_box.x - tracker_box.w / 2, tracker_box.y - tracker_box.h / 2, tracker_box.w, tracker_box.h);
        cv::rectangle(img, rst, cv::Scalar(0,255,0), 2, cv::LINE_8, 0);

        // Draw label background box for better readability on complex scenes.
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        cv::Point text_org(rst.x, rst.y - 4);
        if (text_org.y - text_size.height < 0) text_org.y = rst.y + text_size.height + 4;
        cv::rectangle(img, cv::Point(text_org.x, text_org.y - text_size.height - 4),
                      cv::Point(text_org.x + text_size.width, text_org.y + 4),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(img, buf, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
    // Save one output image per inference call.
    std::string output_name = "./output_" + std::to_string(infer_count_++) + ".jpg";
    cv::imwrite(output_name, img);
    sample::user_visible_stream_log("input image file path: ", this->mParams.inferImageFileName, ", output image:", output_name);
}