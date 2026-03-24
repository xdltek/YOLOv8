#include "calibrator.h"
#include "opencv2/opencv.hpp"
#include <stdio.h>

Int8EntropyCalibrator::Int8EntropyCalibrator() {
    calibration_table_file_path_ = "quant_model.int8.table";
}

Int8EntropyCalibrator::~Int8EntropyCalibrator() {
    if (cache_buf_ != nullptr) {
        free(cache_buf_);
    }
}

const void* Int8EntropyCalibrator::readCalibrationCache(std::size_t& length) {
    FILE *fp = fopen(calibration_table_file_path_.c_str(), "rb");
    if (fp == nullptr) {
        sample::LOG_ERROR() << "Cannot open calibration table file: " << calibration_table_file_path_ << std::endl;

        length = 0;
        return nullptr;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        sample::LOG_ERROR() << "fseek(SEEK_END) failed for: " << calibration_table_file_path_ << std::endl;
        fclose(fp);
        length = 0;
        return nullptr;
    }
    const long f_size_l = ftell(fp);
    if (f_size_l <= 0) {
        sample::LOG_ERROR() << "ftell() failed or empty file: " << calibration_table_file_path_ << std::endl;
        fclose(fp);
        length = 0;
        return nullptr;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        sample::LOG_ERROR() << "fseek(SEEK_SET) failed for: " << calibration_table_file_path_ << std::endl;
        fclose(fp);
        length = 0;
        return nullptr;
    }

    const size_t f_size = static_cast<size_t>(f_size_l);
    cache_buf_ = malloc(f_size);
    if (cache_buf_ == nullptr) {
        sample::LOG_ERROR() << "malloc failed for calibration cache, size=" << f_size << std::endl;
        fclose(fp);
        length = 0;
        return nullptr;
    }

    const size_t n_read = fread(cache_buf_, 1, f_size, fp);
    if (n_read != f_size) {
        sample::LOG_ERROR() << "fread() failed: expected=" << f_size << " actual=" << n_read
                            << " file=" << calibration_table_file_path_ << std::endl;
        free(cache_buf_);
        cache_buf_ = nullptr;
        fclose(fp);
        length = 0;
        return nullptr;
    }

    fclose(fp);

    length = f_size;
    return cache_buf_;
}

bool Int8EntropyCalibrator::getBatch(void **bindings, const char **names, int nbBindings) {

    return true;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void *ptr, std::size_t length) {
    if (ptr == nullptr || length == 0) {
        return;
    }

    FILE *fp = fopen(calibration_table_file_path_.c_str(), "wb");
    if (fp == nullptr) {
        sample::LOG_ERROR() << "Cannot open calibration table file for writing: " << calibration_table_file_path_ << std::endl;
        return;
    }

    const size_t n_written = fwrite(ptr, 1, length, fp);
    if (n_written != length) {
        sample::LOG_ERROR() << "fwrite() failed: expected=" << length << " actual=" << n_written
                            << " file=" << calibration_table_file_path_ << std::endl;
    }
    fclose(fp);
}