#pragma  once
#include <string>

#include "infer/infer_sample_base.h"
#include "opencv2/opencv.hpp"
static int image_index = 0;

struct ClassRes
{
    int classes;
    float prob;
};

struct DetectRes : ClassRes
{
    float x;
    float y;
    float w;
    float h;
};

class Yolov8s: public SampleModel
{
public:
    Yolov8s(const samplesCommon::ModelSampleParams& params);

    bool PostProcess(const samplesCommon::RppBufferManager &buffers);

    bool PreProcess(const samplesCommon::RppBufferManager& buffers);
private:
    int input_width_;
    int input_height_;
    int output_height_;
    int output_width_;
    int infer_count_;

    float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b);

    std::vector<DetectRes> NMS(std::vector<float> output, cv::Mat &img);

    void Draw(cv::Mat &img, std::vector<DetectRes> &result);
};