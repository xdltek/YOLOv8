/** @file parser_api.h
 *
 * @brief ONNX parser helper used by sample engine build flow.
 */


#ifndef RPPRT_PARSER_API_H
#define RPPRT_PARSER_API_H

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
 * @brief Read ONNX file into memory and parse it into network definition.
 * @param onnx_filename ONNX model file path.
 * @param builder Runtime builder handle (kept for API compatibility).
 * @param network Target network to fill.
 * @param parser ONNX parser instance.
 * @return 0 on success, negative error code on failure.
 */
inline int onnx_parser(std::string onnx_filename, infer1::IBuilder *builder,
                infer1::INetworkDefinition *network, onnxparser::IParser *parser)
{
	(void)builder; // Unused in current parser path.
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

#endif // RPPRT_PARSER_API_H
