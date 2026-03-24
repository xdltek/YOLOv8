/** @file sampleCommon.h
 *
 * @brief Shared utility helpers for sample runtime, memory, timing, and file handling.
 */

#ifndef SAMPLES_COMMON_SAMPLECOMMON_H_
#define SAMPLES_COMMON_SAMPLECOMMON_H_

#include <dlfcn.h>
#include <stdio.h>  // fileno
#include <unistd.h> // lockf

#include "logging.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
#include <random>
#include <exception>

#include "Infer.h"
#include "rpp_runtime.h"

using namespace infer1;

#if defined(__aarch64__) || defined(__QNX__)
#define ENABLE_DLA_API 1
#endif

#define CHECK_RETURN_W_MSG(status, val, errMsg)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(status))                                                                                                 \
        {                                                                                                              \
            sample::LOG_ERROR() << errMsg << " Error in " << __FILE__ << ", function " << FN_NAME << "(), line " << __LINE__     \
                      << std::endl;                                                                                    \
            return val;                                                                                                \
        }                                                                                                              \
    } while (0)

#undef ASSERT
#define ASSERT(condition)                                                   \
    do                                                                      \
    {                                                                       \
        if (!(condition))                                                   \
        {                                                                   \
            sample::LOG_ERROR() << "Assertion failure: " << #condition << std::endl;  \
            throw std::runtime_error("Assertion failure");                             \
        }                                                                   \
    } while (0)


#define CHECK_RETURN(status, val) CHECK_RETURN_W_MSG(status, val, "")

constexpr void checkRTError(rtError_t error) {
    if (error == rtError_t::rtSuccess) {
        return;
    }

    sample::LOG_ERROR() << "Rpp Runtime failure: " << error << std::endl;
    std::abort();
}

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator "" _GiB(unsigned long long val) {
    return val * (1 << 30);
}

constexpr long long int operator "" _MiB(unsigned long long val) {
    return val * (1 << 20);
}

constexpr long long int operator "" _KiB(unsigned long long val) {
    return val * (1 << 10);
}


namespace samplesCommon {


class HostMemory {
public:
    HostMemory() = delete;

    virtual void *data() const noexcept {
        return mData;
    }

    virtual std::size_t size() const noexcept {
        return mSize;
    }

    virtual infer1::DataType type() const noexcept {
        return mType;
    }

    virtual ~HostMemory() {}

protected:
    HostMemory(std::size_t size, infer1::DataType type)
            : mData{nullptr}, mSize(size), mType(type) {
    }

    void *mData;
    std::size_t mSize;
    infer1::DataType mType;
};


template<typename ElemType, DataType dataType>
class TypedHostMemory : public HostMemory {
public:
    explicit TypedHostMemory(std::size_t size)
            : HostMemory(size, dataType) {
        mData = new ElemType[size];
    };

    ~TypedHostMemory() noexcept {
        delete[](ElemType *) mData;
    }

    ElemType *raw() noexcept {
        return static_cast<ElemType *>(data());
    }
};

using FloatMemory = TypedHostMemory<float, DataType::kFLOAT>;
using HalfMemory = TypedHostMemory<uint16_t, DataType::kHALF>;
using ByteMemory = TypedHostMemory<uint8_t, DataType::kINT8>;

//! Return vector of indices that puts magnitudes of sequence in descending order.
template<class Iter>
std::vector<size_t> argMagnitudeSort(Iter begin, Iter end) {
    std::vector<size_t> indices(end - begin);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&begin](size_t i, size_t j) { return std::abs(begin[j]) < std::abs(begin[i]); });
    return indices;
}

inline bool readReferenceFile(const std::string &fileName, std::vector<std::string> &refVector) {
    std::ifstream infile(fileName);
    if (!infile.is_open()) {
        std::cout << "ERROR: readReferenceFile: Attempting to read from a file that is not open." << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty())
            continue;
        refVector.push_back(line);
    }
    infile.close();
    return true;
}

template<typename T>
std::vector<std::string> classify(
        const std::vector<std::string> &refVector, const std::vector<T> &output, const size_t topK) {
    const auto inds = samplesCommon::argMagnitudeSort(output.cbegin(), output.cend());
    std::vector<std::string> result;
    result.reserve(topK);
    for (size_t k = 0; k < topK; ++k) {
        result.push_back(refVector[inds[k]]);
    }
    return result;
}

// Ensures that every tensor used by a network has a dynamic range set.
//
// All tensors in a network must have a dynamic range specified if a calibrator is not used.
// This function is just a utility to globally fill in missing scales and zero-points for the entire network.
//
// If a tensor does not have a dyanamic range set, it is assigned inRange or outRange as follows:
//
// * If the tensor is the input to a layer or output of a pooling node, its dynamic range is derived from inRange.
// * Otherwise its dynamic range is derived from outRange.
//
// The default parameter values are intended to demonstrate, for final layers in the network,
// cases where dynamic ranges are asymmetric.
//
// The default parameter values choosen arbitrarily. Range values should be choosen such that
// we avoid underflow or overflow. Also range value should be non zero to avoid uniform zero scale tensor.
inline void setAllDynamicRanges(INetworkDefinition *network, float inRange = 2.0f, float outRange = 4.0f) {
    // Ensure that all layer inputs have a scale.
    for (int i = 0; i < network->getNbLayers(); i++) {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++) {
            ITensor *input{layer->getInput(j)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input != nullptr && !input->dynamicRangeIsSet()) {
                ASSERT(input->setDynamicRange(-inRange, inRange));
            }
        }
    }

    // Ensure that all layer outputs have a scale.
    // Tensors that are also inputs to layers are ingored here
    // since the previous loop nest assigned scales to them.
    for (int i = 0; i < network->getNbLayers(); i++) {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++) {
            ITensor *output{layer->getOutput(j)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet()) {
                // Pooling must have the same input and output scales.
                if (layer->getType() == LayerType::kPOOLING) {
                    ASSERT(output->setDynamicRange(-inRange, inRange));
                } else {
                    ASSERT(output->setDynamicRange(-outRange, outRange));
                }
            }
        }
    }
}

class TimerBase {
public:
    virtual void start() {}

    virtual void stop() {}

    float microseconds() const noexcept {
        return mMs * 1000.f;
    }

    float milliseconds() const noexcept {
        return mMs;
    }

    float seconds() const noexcept {
        return mMs / 1000.f;
    }

    void reset() noexcept {
        mMs = 0.f;
    }

protected:
    float mMs{0.0f};
};

template<typename Clock>
class CpuTimer : public TimerBase {
public:
    using clock_type = Clock;

    void start() {
        mStart = Clock::now();
    }

    void stop() {
        mStop = Clock::now();
        mMs += std::chrono::duration<float, std::milli>{mStop - mStart}.count();
    }

private:
    std::chrono::time_point<Clock> mStart, mStop;
}; // class CpuTimer

using PreciseCpuTimer = CpuTimer<std::chrono::high_resolution_clock>;

inline std::string data_type_to_string(infer1::DataType t) {
    switch (t) {
        case infer1::DataType::kFLOAT:
            return "kFLOAT";
        case infer1::DataType::kHALF:
            return "kHALF";
        case infer1::DataType::kINT8:
            return "kINT8";
        case infer1::DataType::kINT32:
            return "kINT32";
        case infer1::DataType::kUINT32:
            return "kUINT32";
        case infer1::DataType::kBF:
            return "kBF";
        case infer1::DataType::kUINT8:
            return "kUINT8";
        case infer1::DataType::kINT16:
            return "kINT16";
        case infer1::DataType::kUINT16:
            return "kUINT16";
        case infer1::DataType::kBOOL:
            return "kBOOL";
        default:
            return "";
    }
}

inline uint32_t getElementSize(infer1::DataType t) {
    switch (t) {
        case infer1::DataType::kINT32:
            return 4;
        case infer1::DataType::kFLOAT:
            return 4;
        case infer1::DataType::kHALF:
            return 2;
        case infer1::DataType::kINT8:
            return 1;
        default:
            throw std::runtime_error("unsupported data type: " + data_type_to_string(t));
            //return 0;
    }
    //return 0;
}

inline int64_t volume(const infer1::Dims &d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline int count_files(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir))
        return -1; // 路径无效

    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) // 只统计文件，跳过目录
            count++;
    }
    return count;
}

inline bool create_folder(const std::string& file_full_path)
{
    std::filesystem::path full(file_full_path);
    if (full.parent_path().string().empty())
    {
        return true;
    }

    bool result = true;
    if (!std::filesystem::exists(full.parent_path()))
    {
        result = std::filesystem::create_directories(full.parent_path());
    }
    return result;
}

inline float get_mean_square_error(const float* expected_values, const float* actual_values, size_t size,
                                   float& error_acc, float& amp_acc) {

    error_acc = 0.0f;
    amp_acc = 0.0f;
    for(size_t i = 0; i < size; i++) {
        float expected = expected_values[i];
        float actual = actual_values[i];
        float error = expected - actual;

        error_acc += error * error;
        amp_acc += expected * expected;
    }

    return (error_acc / amp_acc);
}

    inline bool file_exists(const std::string& file_path)
    {
        std::ifstream f(file_path.c_str());
        bool exists = f.good();
        f.close();

        return exists;
    }

    inline unsigned int get_random_seed() {
        std::random_device rd;
        return rd();
    }

    inline void fill_random_data(float* ptr, size_t size, unsigned int seed, float min, float max) {
        //std::default_random_engine engine(seed);
        std::mt19937 engine(seed);
        std::uniform_real_distribution<float> distribution(min, max);

        for (size_t i = 0; i < size; i++) {
            ptr[i] = distribution(engine);
        }
    }

    inline void fill_random_data(std::vector<float>& fixed_vector_to_fill, unsigned int seed, float min, float max) {
        fill_random_data(fixed_vector_to_fill.data(), fixed_vector_to_fill.size(), seed, min, max);
    }

    //! Locate path to file, given its filename or filepath suffix and possible dirs it might lie in.
//! Function will also walk back MAX_DEPTH dirs from CWD to check for such a file path.
    inline std::string locateFile(
            const std::string& filepathSuffix, const std::vector<std::string>& directories, bool reportError = true)
    {
        const int MAX_DEPTH{10};
        bool found{false};
        std::string filepath;

        for (auto& dir : directories)
        {
            if (!dir.empty() && dir.back() != '/')
            {
                filepath = dir + "/" + filepathSuffix;
            }
            else
            {
                filepath = dir + filepathSuffix;
            }

            for (int i = 0; i < MAX_DEPTH && !found; i++)
            {
                const std::ifstream checkFile(filepath);
                found = checkFile.is_open();
                if (found)
                {
                    break;
                }

                filepath = "../" + filepath; // Try again in parent dir
            }

            if (found)
            {
                break;
            }

            filepath.clear();
        }

        // Could not find the file
        if (filepath.empty())
        {
            const std::string dirList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
                                                        [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
            std::cout << "Could not find " << filepathSuffix << " in data directories:\n\t" << dirList << std::endl;

            if (reportError)
            {
                std::cout << "&&&& FAILED" << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        return filepath;
    }


} // namespace samplesCommon

extern bool IsGlobalStopwatchEnabled();
extern void SetGlobalStopwatchStatus(bool enable);

extern void StartGlobalStopwatch();
extern void StopGlobalStopwatch();
extern void RecordGlobalStopwatch(const std::string& text);
extern void ResetGlobalStopwatch();
extern bool IsGlobalStopwatchRunning();
extern void ShowStopwatchInMilliseconds();
extern void ShowStopwatchInMicroseconds();


#endif // SAMPLES_COMMON_SAMPLECOMMON_H_
