/**
 * @file perf_trace_session.h
 * @brief Optional rpp_perf trace session helpers.
 */
#ifndef XDLTEK_SAMPLES_PERF_TRACE_SESSION_H
#define XDLTEK_SAMPLES_PERF_TRACE_SESSION_H

#include <string>

/**
 * @brief Start an rpp_perf trace window when the caller enables tracing.
 * @param prefix Prefix used by rpp_perf when naming trace output.
 * @param enabled Caller-controlled switch, usually derived from the build configuration.
 * @param output_dir Directory where trace files should be written.
 */
bool perf_trace_start(const char* prefix, bool enabled, const char* output_dir);

/**
 * @brief Forward RPP driver records into the active trace window.
 * @param mode Driver trace mode passed to rppLogsDumpToTraceWindows.
 */
bool perf_trace_enable_driver(unsigned int mode = 0);

/**
 * @brief Stop forwarding RPP driver records into the active trace window.
 */
void perf_trace_disable_driver();

/**
 * @brief End the active trace window and release global trace state.
 */
void perf_trace_shutdown();

/**
 * @brief Return whether an rpp_perf trace window is active and accepting events.
 */
bool perf_trace_enabled();

/**
 * @brief Begin a named trace scope when tracing is active.
 */
void perf_trace_begin(const char* name, const char* category);

/**
 * @brief End a named trace scope when tracing is active.
 */
void perf_trace_end(const char* name, const char* category);

/**
 * @brief RAII owner for an optional demo-level rpp_perf trace session.
 */
class PerfTraceSession
{
public:
    /**
     * @brief Own a demo-level trace session controlled by build configuration.
     * @param prefix Prefix used by rpp_perf when naming trace output.
     * @param enabled Whether trace capture should be active.
     * @param output_dir Directory where trace files should be written.
     */
    PerfTraceSession(const char* prefix, bool enabled, const std::string& output_dir = "trace")
    {
        active_ = perf_trace_start(prefix, enabled, output_dir.c_str());
    }

    ~PerfTraceSession()
    {
        if (active_) {
            perf_trace_shutdown();
        }
    }

    /**
     * @brief Return whether this session successfully started trace capture.
     */
    bool active() const { return active_; }
    /**
     * @brief Enable driver-level trace records for this active session.
     * @param mode Driver trace mode passed to rppLogsDumpToTraceWindows.
     */
    bool enableDriverTrace(unsigned int mode = 0) const;

    /**
     * @brief Disable copying because a trace session owns process-wide trace state.
     */
    PerfTraceSession(const PerfTraceSession&) = delete;
    /**
     * @brief Disable copy assignment because a trace session owns process-wide trace state.
     */
    PerfTraceSession& operator=(const PerfTraceSession&) = delete;

private:
    bool active_ = false;
};

/**
 * @brief RAII helper that emits one scoped event when rpp_perf tracing is active.
 */
class PerfScope
{
public:
    /**
     * @brief Begin a scoped trace event and close it automatically at scope exit.
     */
    PerfScope(const char* name, const char* category = "yolov8")
        : name_(name), category_(category)
    {
        perf_trace_begin(name_, category_);
    }

    ~PerfScope()
    {
        perf_trace_end(name_, category_);
    }

    /**
     * @brief Disable copying because a scope must end exactly once.
     */
    PerfScope(const PerfScope&) = delete;
    /**
     * @brief Disable copy assignment because a scope must end exactly once.
     */
    PerfScope& operator=(const PerfScope&) = delete;

private:
    const char* name_ = nullptr;
    const char* category_ = nullptr;
};

#define PERF_CONCAT_INNER(a, b) a##b
#define PERF_CONCAT(a, b) PERF_CONCAT_INNER(a, b)
#define PERF_SCOPE(name) PerfScope PERF_CONCAT(perf_scope_, __LINE__)(name)
#define PERF_SCOPE_CATE(name, category) PerfScope PERF_CONCAT(perf_scope_, __LINE__)(name, category)

#endif // XDLTEK_SAMPLES_PERF_TRACE_SESSION_H
