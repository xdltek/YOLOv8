/**
 * @file argsParser.h
 * @brief Parameter structures and legacy command-line parsing helpers for demo samples.
 */

#ifndef RPPRT_ARGS_PARSER_H
#define RPPRT_ARGS_PARSER_H

#include <string>
#include <vector>
#include <getopt.h>
#include <iostream>

namespace samplesCommon
{
    struct ModelSampleParams {
        int32_t batchSize{1};                        //!< Number of inputs in a batch
        bool int8{false};                            //!< Enable INT8 mode when supported by model/runtime.
        bool bf16{true};                             //!< Allow running the network in BF16 mode.
        std::vector<std::string> dataDirs;           //!< Directory paths where sample data files are stored
        std::string onnxFileName;                    //!< Filename of ONNX file of a network
        std::vector<std::string> inputTensorNames;   //!< Network input tensor names.
        std::vector<std::string> outputTensorNames;  //!< Network output tensor names.
        std::string referenceFileName;               //!< Legacy label file path (optional, not used in current demo).
        long long image_count{1};                    //!< Number of images to process.
        long long loops{1};                          //!< Timed inference runs per image.
        std::string inferImageFileName;              //!< Current input image path.
        bool check_inf_nan{false};                   //!< Enable NaN/Inf check on output tensor values.
        std::string model_config;                    //!< Optional runtime model config file.
    };


//!
//! \brief The SampleParams structure groups the basic parameters required by
//!        all sample networks.
//!
    struct SampleParams
    {
        int32_t batchSize{1};              //!< Number of inputs in a batch
        bool int8{false};                  //!< Allow running the network in INT8 mode.
        bool bf16{true};                  //!< Allow running the network in BF16 mode.
        std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
        std::vector<std::string> inputTensorNames;
        std::vector<std::string> outputTensorNames;
    };

//!
//! \brief The CaffeSampleParams structure groups the additional parameters required by
//!         networks that use caffe
//!
    struct CaffeSampleParams : public SampleParams
    {
        std::string prototxtFileName; //!< Filename of prototxt design file of a network
        std::string weightsFileName;  //!< Filename of trained weights file of a network
        std::string meanFileName;     //!< Filename of mean file of a network
    };

//!
//! \brief The OnnxSampleParams structure groups the additional parameters required by
//!         networks that use ONNX
//!
    struct OnnxSampleParams : public SampleParams
    {
        std::string onnxFileName; //!< Filename of ONNX file of a network
    };

//!
//! \brief The UffSampleParams structure groups the additional parameters required by
//!         networks that use Uff
//!
    struct UffSampleParams : public SampleParams
    {
        std::string uffFileName; //!< Filename of uff file of a network
    };

//!
//! \brief The SampleResnetParams structure groups the additional parameters required by
//!         the INT8 API sample
//!
    struct ResnetSampleParams : public OnnxSampleParams
    {
        bool verbose{false};

        std::vector<std::string> dataDirs;
        std::string imageFileName;
        std::string referenceFileName;
        std::string weightsFile;
        std::string calibratorFileName;
        long long loops {1};
        int32_t c;
        int32_t h;
        int32_t w;
        std::string inferImageFileName;
        int thread_count {1};

        //bool writeNetworkTensors{false};
        //std::string dynamicRangeFileName;
        //std::string networkTensorsFileName;

    };

    struct ObjectDetectionParams : public OnnxSampleParams
    {
        bool verbose{false};

        std::string imageFileName;
        std::string classNamesFileName;
        int32_t c;
        int32_t h;
        int32_t w;
//        std::string inferImageFileName;
        int thread_count {1};
        int loops {1};

        //bool writeNetworkTensors{false};
        //std::string dynamicRangeFileName;
        //std::string networkTensorsFileName;

    };


    //!
    //! \brief Struct to maintain command-line arguments.
    //!
    struct Args
    {
        bool runInInt8{false};
        bool runInBf16{true};
        bool help{false};
        int32_t batch{1};
        std::vector<std::string> dataDirs;
        std::string saveEngine;
        std::string loadEngine;
        bool useILoop{false};
    };


//!
//! \brief Populates the Args struct with the provided command-line parameters.
//!
//! \throw invalid_argument if any of the arguments are not valid
//!
//! \return boolean If return value is true, execution can continue, otherwise program should exit
//!
    inline bool parseArgs(Args& args, int32_t argc, char* argv[])
    {
        while (1)
        {
            int32_t arg;
            static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"datadir", required_argument, 0, 'd'},
                                                   {"int8", no_argument, 0, 'i'}, {"bf16", no_argument, 0, 'f'}, {"useILoop", no_argument, 0, 'l'},
                                                   {"saveEngine", required_argument, 0, 's'}, {"loadEngine", no_argument, 0, 'o'}, {"batch", required_argument, 0, 'b'}, {nullptr, 0, nullptr, 0}};
            int32_t option_index = 0;
            arg = getopt_long(argc, argv, "hd:iu", long_options, &option_index);
            if (arg == -1)
            {
                break;
            }

            switch (arg)
            {
                case 'h': args.help = true; return true;
                case 'd':
                    if (optarg)
                    {
                        args.dataDirs.push_back(optarg);
                    }
                    else
                    {
                        std::cerr << "ERROR: --datadir requires option argument" << std::endl;
                        return false;
                    }
                    break;
                case 's':
                    if (optarg)
                    {
                        args.saveEngine = optarg;
                    }
                    break;
                case 'o':
                    if (optarg)
                    {
                        args.loadEngine = optarg;
                    }
                    break;
                case 'i': args.runInInt8 = true; break;
                case 'f': args.runInBf16 = true; break;
                case 'l': args.useILoop = true; break;
                case 'b':
                    if (optarg)
                    {
                        args.batch = std::stoi(optarg);
                    }
                    break;
                default: return false;
            }
        }
        return true;
    }

} // namespace samplesCommon

#endif // RPPRT_ARGS_PARSER_H
