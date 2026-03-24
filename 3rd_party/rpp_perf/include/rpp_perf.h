#ifndef RPP_PERF_H
#define RPP_PERF_H

/** @brief Lightweight C trace collector for Perfetto/Chrome JSON events with optional multi-window support. */

#include <stdint.h>

#ifdef _WIN32
#define BUILDING_DLL
#endif
#ifdef BUILDING_DLL
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT 
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/* Scope name buffer size */
#ifndef RPP_PERF_NAME_SIZE
#define RPP_PERF_NAME_SIZE 128
#endif

/* Category buffer size (for file:line) */
#ifndef RPP_PERF_CAT_SIZE
#define RPP_PERF_CAT_SIZE 128
#endif

/* Structure definition needed for macros */
struct rpp_perf_scope {
    char name[RPP_PERF_NAME_SIZE];
    char cat[RPP_PERF_CAT_SIZE];
    uint32_t pid;
    uint64_t tid;
    uint64_t start_ts_us;
    int active;
};

typedef struct rpp_perf_scope rpp_perf_scope_t;

/** @brief Add a trace window; returns window_id (0 on failure). */
DLL_EXPORT uint32_t rpp_perf_add_trace_window(const char* name, uint32_t event_count);
DLL_EXPORT uint32_t rpp_perf_remove_trace_window(uint32_t window_id);
/** @brief Set active trace window for current thread. */
DLL_EXPORT void rpp_perf_set_active_window(uint32_t window_id);
DLL_EXPORT void rpp_perf_set_thread_name(uint64_t thread_id, const char* name);
DLL_EXPORT void rpp_perf_set_trace_dir(const char* dir);

/** @brief Trace primitive APIs. */
DLL_EXPORT rpp_perf_scope_t rpp_perf_trace_function(const char* func, const char* file, int line);
DLL_EXPORT rpp_perf_scope_t rpp_perf_trace_scope(const char* name, const char* file, int line);
DLL_EXPORT void rpp_perf_trace_value_u64(const char* name, uint64_t value, const char* file, int line);

/** @brief End trace scope (mainly for MSVC/manual scope control). */
DLL_EXPORT void rpp_perf_scope_end(rpp_perf_scope_t* scope);

/** @brief Get current thread ID (cross-platform). */
DLL_EXPORT uint64_t rpp_perf_thread_self(void);

#ifdef __cplusplus
}
#endif

/** @brief Convenience trace macros. */
  #ifndef STRINGIFY
    #define STRINGIFY(x) #x
  #endif
  #ifndef TOSTRING
    #define TOSTRING(x) STRINGIFY(x)
  #endif

  #define TRACE_START(name, N)      (uint32_t)rpp_perf_add_trace_window((name), 1024u * (uint32_t)(N))
  #define TRACE_START_N(name, num)  (uint32_t)rpp_perf_add_trace_window((name), (uint32_t)(num))
  #define TRACE_END(num)  (uint32_t)rpp_perf_remove_trace_window((uint32_t)(num))

  #if defined(__GNUC__) || defined(__clang__)
    #define RPP_PERF_CLEANUP __attribute__((cleanup(rpp_perf_scope_end)))
    #define TRACE_FUNC() \
      rpp_perf_scope_t RPP_PERF_CLEANUP __rpp_perf_scope = rpp_perf_trace_function(__func__, __FILE__, __LINE__)
    #define TRACE_SCOPE_ARGS(name, ...) \
      rpp_perf_scope_t RPP_PERF_CLEANUP __rpp_perf_scope = rpp_perf_trace_scope((name), __FILE__, __LINE__)
    #define TRACE_LAMBDA_ARGS(name, ...) \
      rpp_perf_scope_t RPP_PERF_CLEANUP __rpp_perf_scope = rpp_perf_trace_scope((name), __FILE__, __LINE__)
    #define TRACE_FUNC_ARGS(name, ...) \
      rpp_perf_scope_t RPP_PERF_CLEANUP __rpp_perf_scope = rpp_perf_trace_scope((name), __FILE__, __LINE__)
    #define TRACE_SCOPE_END(scope)
  #else
    /* MSVC: no cleanup attribute, caller may explicitly end scope via TRACE_SCOPE_END(__rpp_perf_scope). */
    #define TRACE_FUNC() \
      rpp_perf_scope_t __rpp_perf_scope = rpp_perf_trace_function(__func__, __FILE__, __LINE__)
    #define TRACE_SCOPE_ARGS(name, ...) \
      rpp_perf_scope_t __rpp_perf_scope = rpp_perf_trace_scope((name), __FILE__, __LINE__)
    #define TRACE_LAMBDA_ARGS(name, ...) \
      rpp_perf_scope_t __rpp_perf_scope = rpp_perf_trace_scope((name), __FILE__, __LINE__)
    #define TRACE_FUNC_ARGS(name, ...) \
      rpp_perf_scope_t __rpp_perf_scope = rpp_perf_trace_scope((name), __FILE__, __LINE__)
    #define TRACE_SCOPE_END(scope) rpp_perf_scope_end(&(scope))
  #endif

  #define TRACE_VALUE_ARGS(value)  rpp_perf_trace_value_u64(STRINGIFY(value), (uint64_t)(value), __FILE__, __LINE__)
  #define TRACE_SET_THREAD_NAME(name) rpp_perf_set_thread_name(rpp_perf_thread_self(), (name))

#endif /* RPP_PERF_H */
