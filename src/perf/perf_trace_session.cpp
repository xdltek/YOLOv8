/**
 * @file perf_trace_session.cpp
 * @brief Optional rpp_perf trace session implementation.
 */
#include "perf_trace_session.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#ifdef YOLO_ENABLE_RPP_PERF
#include <rpp_drv_api.h>
#include <rpp_perf.h>
#include <rpp_runtime.h>
#endif

namespace
{
#ifdef YOLO_ENABLE_RPP_PERF
uint32_t g_trace_window = 0;
bool g_trace_started = false;
bool g_driver_trace_enabled = false;
#endif
}

/**
 * @brief Start a process-wide rpp_perf trace window when enabled by the demo.
 */
bool perf_trace_start(const char* prefix, bool enabled, const char* output_dir)
{
#ifdef YOLO_ENABLE_RPP_PERF
    // Keep one active window per process so nested components share the same trace output.
    if (g_trace_started) {
        return true;
    }
    if (!enabled) {
        return false;
    }

    // Create and resolve the requested trace directory before handing it to rpp_perf.
    std::string dir = (output_dir == nullptr || output_dir[0] == '\0') ? "trace" : output_dir;
    std::error_code error;
    std::filesystem::path dir_path(dir);
    std::filesystem::create_directories(dir_path, error);
    if (error) {
        std::cerr << "Failed to create rpp_perf trace directory: " << dir
                  << ", error=" << error.message() << std::endl;
        return false;
    }
    std::filesystem::path absolute_dir = std::filesystem::absolute(dir_path, error);
    if (error) {
        std::cerr << "Failed to resolve rpp_perf trace directory: " << dir
                  << ", error=" << error.message() << std::endl;
        return false;
    }

    // Match the OpenRT sandbox behavior: validate writability before trace start.
    std::filesystem::path probe_file = absolute_dir / ".yolov8_demo_trace_write_probe";
    {
        std::ofstream probe_stream(probe_file.string(), std::ios::out | std::ios::trunc);
        if (!probe_stream.is_open()) {
            std::cerr << "rpp_perf trace directory is not writable: " << absolute_dir.string() << std::endl;
            return false;
        }
    }
    std::filesystem::remove(probe_file, error);

    // TRACE_START opens the Perfetto window; later PERF_SCOPE calls add events to it.
    rpp_perf_set_trace_dir(absolute_dir.string().c_str());
    g_trace_window = TRACE_START(prefix == nullptr ? "yolov8_demo" : prefix);
    if (g_trace_window == 0) {
        std::cerr << "Failed to start rpp_perf trace." << std::endl;
        return false;
    }

    g_trace_started = true;
    const char* path = rpp_perf_win_path(g_trace_window);
    if (path != nullptr) {
        std::cout << "rpp_perf trace: " << path << std::endl;
    }
    return true;
#else
    (void)prefix;
    (void)output_dir;
    if (enabled) {
        std::cerr << "rpp_perf trace was requested, but this build was configured without "
                  << "YOLO_ENABLE_RPP_PERF. Reconfigure with "
                  << "`cmake .. -DYOLO_ENABLE_RPP_PERF=ON` to enable trace capture."
                  << std::endl;
    }
    return false;
#endif
}

/**
 * @brief Attach RPP driver records to the active rpp_perf trace window.
 */
bool perf_trace_enable_driver(unsigned int mode)
{
#ifdef YOLO_ENABLE_RPP_PERF
    // Driver records are useful only after a trace window exists.
    if (!g_trace_started || g_trace_window == 0) {
        return false;
    }
    if (g_driver_trace_enabled) {
        return true;
    }

    // OpenRT sandbox initializes the driver before attaching driver/device logs.
    int device_count = 0;
    rtError_t status = rtGetDeviceCount(&device_count);
    if (status != rtSuccess) {
        std::cerr << "Failed to initialize RPP driver for trace, rtError="
                  << static_cast<int>(status) << std::endl;
        return false;
    }

    // rppLogsDumpToTraceWindows streams device/runtime records into this demo trace window.
    RPPresult result = rppLogsDumpToTraceWindows(g_trace_window, mode);
    if (result != RPP_SUCCESS) {
        std::cerr << "Failed to enable RPP driver trace, RPPresult="
                  << static_cast<int>(result) << std::endl;
        return false;
    }

    g_driver_trace_enabled = true;
    return true;
#else
    (void)mode;
    return false;
#endif
}

/**
 * @brief Detach RPP driver records from the active trace window.
 */
void perf_trace_disable_driver()
{
#ifdef YOLO_ENABLE_RPP_PERF
    // Passing RPP_INVALID_WID detaches driver logging from the trace window.
    if (!g_driver_trace_enabled) {
        return;
    }
    RPPresult result = rppLogsDumpToTraceWindows(RPP_INVALID_WID, 0);
    if (result != RPP_SUCCESS) {
        std::cerr << "Failed to disable RPP driver trace, RPPresult="
                  << static_cast<int>(result) << std::endl;
    }
    g_driver_trace_enabled = false;
#endif
}

/**
 * @brief Close the active trace window and reset process-wide trace state.
 */
void perf_trace_shutdown()
{
#ifdef YOLO_ENABLE_RPP_PERF
    // Stop driver forwarding first so TRACE_END_ALL closes a complete trace.
    if (!g_trace_started) {
        return;
    }
    perf_trace_disable_driver();
    TRACE_END_ALL();
    g_trace_started = false;
    g_trace_window = 0;
#endif
}

/**
 * @brief Return true when rpp_perf currently accepts scoped events.
 */
bool perf_trace_enabled()
{
#ifdef YOLO_ENABLE_RPP_PERF
    return g_trace_started && _rpp_perf_enabled(g_trace_window);
#else
    return false;
#endif
}

/**
 * @brief Begin one named rpp_perf scope if tracing is active.
 */
void perf_trace_begin(const char* name, const char* category)
{
#ifdef YOLO_ENABLE_RPP_PERF
    // Component code can call this unconditionally; it becomes a no-op without an active trace window.
    if (perf_trace_enabled()) {
        TRACE_FUNC_CATE(name, category == nullptr ? "yolov8" : category);
    }
#else
    (void)name;
    (void)category;
#endif
}

/**
 * @brief End one named rpp_perf scope if tracing is active.
 */
void perf_trace_end(const char* name, const char* category)
{
#ifdef YOLO_ENABLE_RPP_PERF
    // Close the matching trace scope only when the trace window is still active.
    if (perf_trace_enabled()) {
        TRACE_FUNC_END_CATE(name, category == nullptr ? "yolov8" : category);
    }
#else
    (void)name;
    (void)category;
#endif
}

/**
 * @brief Enable driver-level trace forwarding for this session.
 */
bool PerfTraceSession::enableDriverTrace(unsigned int mode) const
{
    return active_ && perf_trace_enable_driver(mode);
}
