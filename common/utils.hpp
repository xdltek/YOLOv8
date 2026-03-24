/**
 * @file utils.hpp
 * @brief Utility helpers for numeric checks and CLI argument preprocessing.
 */
#pragma once
#include <vector>
#include <iostream>
#include <math.h>
#include <numeric>
template<class T>
float ErrMaxMse(std::vector<T> ort, std::vector<T> rt)
{
    if(ort.size() != rt.size())
    {
        std::cout << "rt and ort data size is not equal" << std::endl;
        return 0.0;
    }

    float err_max =0.0;
    int pos = 0;
    for(int i = 0; i < ort.size(); ++i)
    {
        if(err_max < std::fabs(ort[i] - rt[i]))
        {
            err_max = std::fabs(ort[i] - rt[i]);
            pos = i;
        }
    }

    float err_max_relative_error = std::fabs(ort[pos] - rt[pos]) / (std::fabs(ort[pos]) + std::fabs(rt[pos]));
    return err_max_relative_error;
}

template<class T>
float CalcSTD(std::vector<T> data)
{
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();

    double variance = 0.0;
    for (float x : data){
        variance += std::pow(x - mean, 2);
    }
    return std::sqrt(variance / data.size());
}

template<typename T>
std::vector<float> PrecisionCheck(std::vector<T>& ort, std::vector<T>& rt)
{
    std::vector<float> res;
    float err_acc = 0;
    float amp_acc = 0;
    std::vector<float> abs_err;
    for (size_t j = 0; j < ort.size(); j++)
    {
        float err = ort.at(j) -rt.at(j);
        err_acc += err*err;
        amp_acc += ort.at(j) * rt.at(j);
        abs_err.emplace_back(std::fabs(err));
    }
    float mse = err_acc / amp_acc;
    float std = CalcSTD(abs_err);
    float err_max_mse = ErrMaxMse(ort, rt);
    res.emplace_back(mse);
    res.emplace_back(std);
    res.emplace_back(err_max_mse);
    res.emplace_back(err_acc);
    res.emplace_back(amp_acc);
    return res;
}

//std::map<std::string, int> GetModelNameAndType(std::string model_name, std::string model_path)
//{
//    Ort::Env env;
//    Ort::Session session_{ env, model_path.c_str() , Ort::SessionOptions{nullptr} };
//    auto type_info = session_.GetInputTypeInfo(0);
//    size_t inputNodeCount = session_.GetInputCount();
//    Ort::AllocatorWithDefaultOptions allocator;
//    std::map<std::string, int>name_2_type;
//    for (size_t i = 0; i < inputNodeCount; ++i)
//    {
//        Ort::TypeInfo inputTypeInfo = session_.GetInputTypeInfo(i);
//        auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
//        ONNXTensorElementDataType inputNodeDataType = input_tensor_info.GetElementType();
////            std::cout << "name " << session_.GetInputNameAllocated(i,allocator).get() << " type: " << inputNodeDataType << std::endl;
//        name_2_type[session_.GetInputNameAllocated(i,allocator).get()] = inputNodeDataType;
//    }
//    size_t outputNodeCount = session_.GetOutputCount();
//    for (size_t i = 0; i < outputNodeCount; ++i)
//    {
//        Ort::TypeInfo outputTypeInfo = session_.GetOutputTypeInfo(i);
//        auto output_tensor_info = outputTypeInfo.GetTensorTypeAndShapeInfo();
//        ONNXTensorElementDataType outputNodeDataType = output_tensor_info.GetElementType();
////            std::cout << "name " << session_.GetOutputNameAllocated(i,allocator).get() << " type: " << outputNodeDataType << std::endl;
//        name_2_type[session_.GetOutputNameAllocated(i,allocator).get()] = outputNodeDataType;
//    }
//    return name_2_type;
//}

template <class T1, class T2>
void castData(std::vector<T1> &in, std::vector<T2> &out){
    int length = in.size();
    out.resize(length);
    for(auto i=0;i<length;i++)
        out[i] = (T2)in[i];
}

inline std::string expand_user_path(const std::string& path) {
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) {
            return home + path.substr(1);
        }
    }
    return path;
}

inline std::vector<std::string> preprocess_args(int argc, char* argv[]) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Handle short argument style with '=' (example: -o=model.onnx).
        if (arg.size() > 2 && arg[0] == '-' && arg[1] != '-' && arg.find('=') != std::string::npos) {
            size_t pos = arg.find('=');
            std::string flag = arg.substr(0, pos);
            std::string value = arg.substr(pos + 1);

            args.push_back(flag);
            args.push_back(value);
        } else {
            args.push_back(arg);
        }
    }
    return args;
}


inline void to_char_argument_vector(const std::vector<std::string>& preprocessed_arguments, char* original_argv[],
                             std::vector<const char*>& fixed_arguments) {
    fixed_arguments.clear();
    fixed_arguments.push_back(original_argv[0]); // Program name.
    for (auto& preprocessed_argument : preprocessed_arguments) {
        fixed_arguments.push_back(preprocessed_argument.data());
    }
}
