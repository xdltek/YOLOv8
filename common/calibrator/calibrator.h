/**
 * @file calibrator.h
 * @brief INT8 entropy calibrator declaration for sample runtime.
 */
#pragma once

#include <stdio.h>
#include <string>

#include "Infer.h"
#include "logging.h"
#include <iostream>

class Int8EntropyCalibrator : public infer1::IInt8EntropyCalibrator2 {
private:
    int batch_size_ = 10;
    std::vector<std::string> input_file_paths_;
    void *cache_buf_ = nullptr;
    int w_ = 0;
    int h_ = 0;
    int c_ = 0;
    std::string calibration_table_file_path_;
    std::string model_file_name_;

public:
    /**
     * @brief Construct calibrator instance.
     */
    Int8EntropyCalibrator();

    /**
     * @brief Release calibrator resources.
     */
    ~Int8EntropyCalibrator() override;

    /**
     * @brief Get calibration batch size.
     * @return Batch size used during calibration.
     */
    int getBatchSize() const override {
        return batch_size_;
    }

    /**
     * @brief Provide one calibration batch to runtime.
     * @param bindings Device bindings array to fill.
     * @param names Binding names array.
     * @param nbBindings Number of bindings.
     * @return true when a batch is provided, false when calibration data ends.
     */
    bool getBatch(void **bindings, const char **names, int nbBindings) ORTNOEXCEPT override;

    /**
     * @brief Read serialized calibration cache.
     * @param length Output cache length.
     * @return Pointer to cache data or nullptr.
     */
    const void* readCalibrationCache(std::size_t& length) ORTNOEXCEPT override;

    /**
     * @brief Persist serialized calibration cache.
     * @param ptr Cache data pointer.
     * @param length Cache byte length.
     */
    void writeCalibrationCache(const void *ptr, std::size_t length) ORTNOEXCEPT override;
};
