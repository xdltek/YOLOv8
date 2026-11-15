/**
 * @file demo_reporter.h
 * @brief Customer-facing console reporting helpers for YOLOv8 applications.
 */
#pragma once

#include "yolo_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace demo_reporter
{
using Clock = std::chrono::steady_clock;

inline bool color_enabled()
{
    const char* setting = std::getenv("YOLO_COLOR");
    return setting == nullptr || std::string(setting) != "never";
}

inline const char* color(const char* code)
{
    return color_enabled() ? code : "";
}

constexpr const char* kReset = "\033[0m";
constexpr const char* kBold = "\033[1m";
constexpr const char* kBlue = "\033[1;34m";
constexpr const char* kCyan = "\033[1;36m";
constexpr const char* kGreen = "\033[1;32m";
constexpr const char* kYellow = "\033[1;33m";

inline void print_stage(const std::string& state, const std::string& message)
{
    const char* stage_color = state == "WARMUP" ? kYellow : kCyan;
    std::cout << color(stage_color) << "[" << std::left << std::setw(8) << state << "]"
              << color(kReset) << " " << message << std::right << std::endl;
}

inline void print_progress(int completed, int total, const Clock::time_point& start)
{
    int interval = total <= 10 ? 1 : std::max(1, total / 10);
    if (completed != 1 && completed != total && (completed % interval) != 0) {
        return;
    }

    constexpr int kBarWidth = 30;
    double fraction = static_cast<double>(completed) / static_cast<double>(total);
    int filled = static_cast<int>(fraction * kBarWidth);
    double elapsed_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    double eta_seconds = completed > 0 ? elapsed_seconds * (total - completed) / completed : 0.0;

    std::cout << color(kCyan) << "[RUNNING ]" << color(kReset) << " [";
    for (int i = 0; i < kBarWidth; ++i) {
        if (i == 0) {
            std::cout << color(completed == total ? kGreen : kBlue);
        }
        std::cout << (i < filled ? '=' : (i == filled && completed < total ? '>' : ' '));
    }
    std::cout << color(kReset) << "] " << color(kBold) << std::setw(3)
              << static_cast<int>(fraction * 100.0) << "%" << color(kReset) << "  "
              << completed << "/" << total << " frames"
              << "  elapsed " << std::fixed << std::setprecision(1) << elapsed_seconds << "s"
              << "  ETA " << eta_seconds << "s" << std::endl;
}

inline void print_summary(const YoloV8StageTimes& times,
                          int loop_count,
                          size_t detection_count,
                          const std::string& output_path)
{
    double end_to_end_fps = times.end_to_end_ms > 0.0 ? 1000.0 / times.end_to_end_ms : 0.0;
    double inference_fps = times.inference_ms > 0.0 ? 1000.0 / times.inference_ms : 0.0;
    double total_ms = times.end_to_end_ms * static_cast<double>(loop_count);
    double input_kib = static_cast<double>(times.input_h2d_bytes) / 1024.0;
    double output_kib = static_cast<double>(times.output_d2h_bytes) / 1024.0;

    std::cout << std::fixed << std::setprecision(3) << color(kBlue)
              << "\n+------------------------------------------------------------------+\n"
              << "|                 YOLOv8 PERFORMANCE REPORT                       |\n"
              << "+------------------------------------------------------------------+\n" << color(kReset)
              << color(kBold) << color(kCyan) << "  PERFORMANCE HIGHLIGHTS\n" << color(kReset)
              << color(kBold) << "  End-to-end FPS           : " << color(kGreen)
              << end_to_end_fps << " FPS" << color(kReset) << "\n"
              << color(kBold) << "  OpenRT inference FPS     : " << color(kGreen)
              << inference_fps << " FPS" << color(kReset) << "\n"
              << color(kBold) << "\n  MEASUREMENT DETAILS\n" << color(kReset)
              << "  End-to-end latency       : " << color(kGreen) << times.end_to_end_ms
              << " ms / frame" << color(kReset) << "\n"
              << "  Measured frames          : " << loop_count << " (+ 1 warmup)\n"
              << "  Benchmark duration       : " << total_ms << " ms\n"
              << "  Detections (last frame)  : " << detection_count << "\n"
              << color(kBold) << "\n  PIPELINE BREAKDOWN (average per frame)\n" << color(kReset)
              << "  Input H2D transfer       : " << times.input_h2d_ms << " ms  (" << input_kib << " KiB)\n"
              << "  Preprocess               : " << times.preprocess_ms << " ms\n"
              << "  OpenRT inference         : " << times.inference_ms << " ms\n"
              << "  Postprocess              : " << times.postprocess_ms << " ms\n"
              << "  Output D2H transfer      : " << times.output_d2h_ms << " ms  (" << output_kib << " KiB)\n"
              << color(kBold) << "\n  RESULT\n" << color(kReset)
              << "  Visualization            : " << output_path << "\n"
              << color(kBlue) << "+------------------------------------------------------------------+\n" << color(kReset)
              << color(kGreen) << "[SUCCESS ]" << color(kReset)
              << " Inference completed successfully." << std::endl;
}
}  // namespace demo_reporter
