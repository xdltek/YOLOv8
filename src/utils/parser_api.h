/** @file parser_api.h
 *
 * @brief Helper for loading and parsing ONNX model files into an OpenRT network.
 * @author XDLTek Technologies
 * COPYRIGHT(c) 2020-2022 XDLTek Technologies.
 * ALL RIGHTS RESERVED	
 *
 * This is Unpublished Proprietary Source Code of XDLTek Technologies
 */


#ifndef OPENRT_PARSER_API_H
#define OPENRT_PARSER_API_H

#include <fstream>
#include <getopt.h>
#include <unistd.h> // For ::getopt
#include <iostream>
#include <ctime>
#include <fcntl.h> // For ::open
#include <limits>

#include "OnnxParser.h"
#include "logging.h"

/**
 * @brief Read an ONNX file from disk and feed it to the provided parser.
 *
 * @return 0 on success, or a negative error code when the file or parser fails.
 */
inline int onnx_parser(std::string onnx_filename, infer1::IBuilder *builder,
                infer1::INetworkDefinition *network, onnxparser::IParser *parser)
{
	std::ifstream onnx_file(onnx_filename.c_str(),
		std::ios::binary | std::ios::ate);
	std::streamsize file_size = onnx_file.tellg();
	onnx_file.seekg(0, std::ios::beg);
	std::vector<char> onnx_buf(file_size);
	if (!onnx_file.read(onnx_buf.data(), onnx_buf.size())) {
        sample::LOG_ERROR() << "ERROR: Failed to read from file: " << onnx_filename << std::endl;
		return -4;
	}
	if (!parser->parse(onnx_buf.data(), onnx_buf.size())) {
		int nerror = parser->getNbErrors();
		for (int i = 0; i < nerror; ++i) {
			onnxparser::IParserError const* error = parser->getError(i);
            sample::LOG_ERROR() << "ERROR: "
				<< error->file() << ":" << error->line()
				<< " In function " << error->func() << ":\n"
				<< "[" << static_cast<int>(error->code()) << "] " << error->desc()
				<< std::endl;
		}

		if(nerror != (int) onnxparser::ErrorCode::kSUCCESS)
        {
            return -1;
        }
		return -5;
	}
	return 0;
}

#endif // OPENRT_PARSER_API_H
