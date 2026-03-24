/**
 * @file rpp_perf.c
 * @brief C implementation of the rpp_perf trace backend.
 */
#include "rpp_perf.h"

/* Define POSIX feature macros before including system headers. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>

/* Cross-platform includes */
#ifdef _WIN32
    #include <windows.h>
    #include <process.h>
    #include <direct.h>
    #include <io.h>
    #include <errno.h>
    #include <fcntl.h>
    #define PATH_SEP '\\'
    #define PATH_SEP_STR "\\"
#else
    #include <pthread.h>
    #include <unistd.h>
    #include <time.h>
    #include <sys/time.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <errno.h>
    #include <fcntl.h>
    #include <signal.h>
    #define PATH_SEP '/'
    #define PATH_SEP_STR "/"
#endif

/* Cross-platform thread local storage */
#ifdef _WIN32
    #define THREAD_LOCAL __declspec(thread)
#else
    #define THREAD_LOCAL __thread
#endif

/* Cross-platform mutex, condition variable and once */
#ifdef _WIN32
    typedef CRITICAL_SECTION rpp_perf_mutex_t;
    typedef CONDITION_VARIABLE rpp_perf_cond_t;
    typedef INIT_ONCE rpp_perf_once_t;
    #define RPP_PERF_ONCE_INIT INIT_ONCE_STATIC_INIT
    static inline void rpp_perf_mutex_init(rpp_perf_mutex_t* m) {
        InitializeCriticalSection(m);
    }
    static inline void rpp_perf_mutex_lock(rpp_perf_mutex_t* m) {
        EnterCriticalSection(m);
    }
    static inline void rpp_perf_mutex_unlock(rpp_perf_mutex_t* m) {
        LeaveCriticalSection(m);
    }
    static inline void rpp_perf_mutex_destroy(rpp_perf_mutex_t* m) {
        DeleteCriticalSection(m);
    }
    static inline void rpp_perf_cond_init(rpp_perf_cond_t* c) {
        InitializeConditionVariable(c);
    }
    static inline void rpp_perf_cond_wait(rpp_perf_cond_t* c, rpp_perf_mutex_t* m) {
        SleepConditionVariableCS(c, m, INFINITE);
    }
    /* Timed condition wait: return 0 on success, non-zero on timeout/error. */
    static inline int rpp_perf_cond_wait_timeout(rpp_perf_cond_t* c, rpp_perf_mutex_t* m, uint32_t timeout_sec) {
        DWORD timeout_ms = (timeout_sec > 0) ? (timeout_sec * 1000) : INFINITE;
        BOOL result = SleepConditionVariableCS(c, m, timeout_ms);
        return result ? 0 : (GetLastError() == ERROR_TIMEOUT ? 1 : -1);
    }
    static inline void rpp_perf_cond_signal(rpp_perf_cond_t* c) {
        WakeConditionVariable(c);
    }
    static inline void rpp_perf_cond_destroy(rpp_perf_cond_t* c) {
        (void)c; /* Windows condition variables don't need explicit destruction */
    }
    static BOOL CALLBACK rpp_perf_init_once_callback(PINIT_ONCE InitOnce, PVOID Parameter, PVOID* lpContext) {
        (void)InitOnce;
        (void)lpContext;
        void (*init_func)(void) = (void (*)(void))Parameter;
        if (init_func) {
            init_func();
        }
        return TRUE;
    }
    static inline void rpp_perf_once(rpp_perf_once_t* once, void (*init_func)(void)) {
        PVOID lpContext = NULL;
        InitOnceExecuteOnce(once, rpp_perf_init_once_callback, (PVOID)init_func, &lpContext);
    }
#else
    typedef pthread_mutex_t rpp_perf_mutex_t;
    typedef pthread_cond_t rpp_perf_cond_t;
    typedef pthread_once_t rpp_perf_once_t;
    #define RPP_PERF_ONCE_INIT PTHREAD_ONCE_INIT
    static inline void rpp_perf_mutex_init(rpp_perf_mutex_t* m) {
        pthread_mutex_init(m, NULL);
    }
    static inline void rpp_perf_mutex_lock(rpp_perf_mutex_t* m) {
        pthread_mutex_lock(m);
    }
    static inline void rpp_perf_mutex_unlock(rpp_perf_mutex_t* m) {
        pthread_mutex_unlock(m);
    }
    static inline void rpp_perf_mutex_destroy(rpp_perf_mutex_t* m) {
        pthread_mutex_destroy(m);
    }
    static inline void rpp_perf_cond_init(rpp_perf_cond_t* c) {
        pthread_cond_init(c, NULL);
    }
    static inline void rpp_perf_cond_wait(rpp_perf_cond_t* c, rpp_perf_mutex_t* m) {
        pthread_cond_wait(c, m);
    }
    /* Timed condition wait: return 0 on success, non-zero on timeout/error. */
    static inline int rpp_perf_cond_wait_timeout(rpp_perf_cond_t* c, rpp_perf_mutex_t* m, uint32_t timeout_sec) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += timeout_sec;
        int result = pthread_cond_timedwait(c, m, &ts);
        return (result == ETIMEDOUT) ? 1 : ((result == 0) ? 0 : -1);
    }
    static inline void rpp_perf_cond_signal(rpp_perf_cond_t* c) {
        pthread_cond_signal(c);
    }
    static inline void rpp_perf_cond_destroy(rpp_perf_cond_t* c) {
        pthread_cond_destroy(c);
    }
    static inline void rpp_perf_once(rpp_perf_once_t* once, void (*init_func)(void)) {
        pthread_once(once, init_func);
    }
#endif

/* ----------------------------------------------------------------------------
 * Design notes:
 * - Multi-window support: each window has an independent event buffer and trace file
 * - Event collection runs only when a trace window is active (reduce runtime overhead)
 * - Global buffers are protected by mutex; TLS batch buffer reduces lock contention
 * - Output format is Chrome Trace Event JSON (.trace), directly importable by Perfetto/Chrome
 * - Long-running trace support via ring-buffer style management and periodic flush
 * - Asynchronous writer thread avoids blocking the main execution path
 * ---------------------------------------------------------------------------- */

typedef enum {
    RPP_PERF_EVT_B = 'B', /* Duration begin */
    RPP_PERF_EVT_E = 'E', /* Duration end */
    RPP_PERF_EVT_C = 'C', /* Counter */
} rpp_perf_evt_type_e;

typedef struct {
    char name[RPP_PERF_NAME_SIZE];
    char cat[RPP_PERF_CAT_SIZE]; /* "file:line" */
    rpp_perf_evt_type_e ph;
    uint64_t ts_us;
    uint32_t pid;
    uint64_t tid;
    uint64_t value;  /* for counter */
} rpp_perf_event_t;

/* Per-window trace state. */
#define RPP_PERF_MAX_WINDOWS 32
#define RPP_PERF_MAX_EVENTS 65536
#define RPP_PERF_WRITE_BATCH_SIZE 4096
#define RPP_PERF_FLUSH_THRESHOLD (RPP_PERF_MAX_EVENTS / 2)

typedef struct {
    uint32_t window_id;
    char name[64];
    uint32_t remaining;
    rpp_perf_event_t events[RPP_PERF_MAX_EVENTS];
    uint32_t event_count;
    uint32_t event_write_pos;
    int pending_write;
    int active; /* Whether this window is active. */
} rpp_perf_window_t;

/* Global state */
static rpp_perf_mutex_t g_mu;
static rpp_perf_cond_t g_write_cond;
static int g_mu_initialized = 0;
static int g_cond_initialized = 0;
static rpp_perf_once_t g_once = RPP_PERF_ONCE_INIT;
static char g_trace_dir[512] = {0};
static int g_shutdown = 0;
static int g_write_thread_running = 0;
#ifdef _WIN32
static HANDLE g_write_thread_handle = NULL;
#else
static pthread_t g_write_thread = 0;
#endif

/* Window management */
static rpp_perf_window_t g_windows[RPP_PERF_MAX_WINDOWS];
static uint32_t g_window_mask = 0;

/* Simple thread-name map (fixed, linear scan) */
#define RPP_PERF_MAX_THREAD_NAMES 256
typedef struct {
    uint64_t tid;
    char name[32];
} rpp_perf_thread_name_t;
static rpp_perf_thread_name_t g_thread_names[RPP_PERF_MAX_THREAD_NAMES];
static uint32_t g_thread_name_count = 0;

/* TLS: per-thread active window selection. */
static THREAD_LOCAL uint32_t g_tls_active_window_id = 0;

/* TLS batch buffer */
#define RPP_PERF_TLS_BATCH 128
static THREAD_LOCAL rpp_perf_event_t g_tls_events[RPP_PERF_TLS_BATCH];
static THREAD_LOCAL uint32_t g_tls_count = 0;

/* Forward declarations */
static void write_trace_file_batch(uint32_t window_id, const char* window_name, 
                                   const rpp_perf_event_t* events, 
                                   uint32_t count, int append,
                                   const rpp_perf_thread_name_t* thread_names_snapshot, 
                                   uint32_t thread_name_count_snapshot);
static void write_trace_file_locked(uint32_t window_id, const char* window_name);
void rpp_perf_flush_on_exit(void);

/** @brief Find an active window slot by window ID (mutex must be held). */
static rpp_perf_window_t* find_window_locked(uint32_t window_id)
{
    rpp_perf_window_t* window = NULL;
    if (g_window_mask & (0x01 << (window_id - 1)))
    {
        window = &g_windows[window_id - 1];
    }
    return window;
}

/** @brief Get the current thread's active window (mutex must be held). */
static rpp_perf_window_t* get_active_window_locked(void)
{
    uint32_t window_id = g_tls_active_window_id;
    if (window_id == 0) {
        /* If no active window is set, return NULL (do not auto-select a window). */
        return NULL;
    }
    return find_window_locked(window_id);
}

/** @brief Background writer thread that drains pending window buffers to disk. */
#ifdef _WIN32
static DWORD WINAPI write_thread_func(LPVOID arg)
#else
static void* write_thread_func(void* arg)
#endif
{
    (void)arg;
    
    for (;;) {
        rpp_perf_mutex_lock(&g_mu);
        
        /* Find one window with pending data to flush. */
        rpp_perf_window_t* window_to_write = NULL;
        for (uint32_t i = 0; i < RPP_PERF_MAX_WINDOWS; i++) {
            if (g_windows[i].active && g_windows[i].event_count > 0) {
                /* During shutdown, flush all remaining data regardless of remaining/pending flags. */
                if (g_shutdown) {
                    window_to_write = &g_windows[i];
                    break;
                }
                /* In normal state, only flush when pending_write is set and remaining > 0. */
                if (g_windows[i].pending_write && g_windows[i].remaining > 0) {
                    window_to_write = &g_windows[i];
                    break;
                }
            }
        }
        
        /* If no pending data exists, wait for signal or shutdown. */
        if (!window_to_write) {
            if (g_shutdown) {
                /* On shutdown and no pending data, exit thread. */
                g_write_thread_running = 0;
                rpp_perf_mutex_unlock(&g_mu);
                break;
            }
            /* Timed wait (5s) to avoid dead-wait in abnormal conditions. */
            int wait_result = rpp_perf_cond_wait_timeout(&g_write_cond, &g_mu, 5);
            if (wait_result == 1) {
                /* Timed out while waiting for pending data. */
                printf("[rpp_perf] Write thread wait timeout (5s), no pending data\n");
            }
            rpp_perf_mutex_unlock(&g_mu);
            continue;
        }
        
        /* Prepare a batch to write. */
        uint32_t write_count = window_to_write->event_count;
        if (write_count > RPP_PERF_WRITE_BATCH_SIZE) {
            write_count = RPP_PERF_WRITE_BATCH_SIZE;
        }
        
        /* Copy events and thread-name map snapshot to minimize lock hold time. */
        rpp_perf_event_t* events_to_write = (rpp_perf_event_t*)malloc(write_count * sizeof(rpp_perf_event_t));
        rpp_perf_thread_name_t* thread_names_snapshot = NULL;
        uint32_t thread_name_count_snapshot = 0;
        uint32_t window_id = window_to_write->window_id;
        char window_name[64];
        (void)snprintf(window_name, sizeof(window_name), "%s", window_to_write->name);
        
        if (events_to_write) {
            uint32_t start_pos = (window_to_write->event_count == RPP_PERF_MAX_EVENTS) ? 
                                 window_to_write->event_write_pos : 0;
            for (uint32_t i = 0; i < write_count; i++) {
                uint32_t idx = (start_pos + i) % RPP_PERF_MAX_EVENTS;
                events_to_write[i] = window_to_write->events[idx];
            }
            
            /* Copy thread-name mapping snapshot for async writer usage. */
            if (g_thread_name_count > 0) {
                thread_names_snapshot = (rpp_perf_thread_name_t*)malloc(
                    g_thread_name_count * sizeof(rpp_perf_thread_name_t));
                if (thread_names_snapshot) {
                    for (uint32_t i = 0; i < g_thread_name_count; i++) {
                        thread_names_snapshot[i] = g_thread_names[i];
                    }
                    thread_name_count_snapshot = g_thread_name_count;
                } else {
                    printf("[rpp_perf] Failed to allocate thread name snapshot memory, thread name mapping will not be used\n");
                }
            }
            
            /* Update window buffer state after copying this batch. */
            if (write_count >= window_to_write->event_count) {
                window_to_write->event_count = 0;
                window_to_write->event_write_pos = 0;
                window_to_write->pending_write = 0;
            } else {
                /* Partial flush: advance write position and keep pending state. */
                window_to_write->event_write_pos = (start_pos + write_count) % RPP_PERF_MAX_EVENTS;
                window_to_write->event_count -= write_count;
                window_to_write->pending_write = (window_to_write->event_count > 0) ? 1 : 0;
            }
            
            /* Append mode: writer handles non-existent file and can downgrade to create mode.
             * Keep append=1 because partial/final batches should continue writing to same trace file. */
            int append = 1;
            
            rpp_perf_mutex_unlock(&g_mu);
            
            /* Perform actual file write outside lock; pass thread-name snapshot. */
            write_trace_file_batch(window_id, window_name, events_to_write, write_count, append,
                                   thread_names_snapshot, thread_name_count_snapshot);
            
            free(events_to_write);
            if (thread_names_snapshot) {
                free(thread_names_snapshot);
            }
            
            rpp_perf_mutex_lock(&g_mu);
        } else {
            /* Allocation failed: fall back to synchronous full flush. */
            printf("[rpp_perf] Memory allocation failed, using synchronous write mode\n");
            window_to_write->pending_write = 0;
            rpp_perf_mutex_unlock(&g_mu);
            write_trace_file_locked(window_id, window_name);
            rpp_perf_mutex_lock(&g_mu);
        }
        
        rpp_perf_mutex_unlock(&g_mu);
    }
    
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

/** @brief Start the asynchronous writer thread if it is not running. */
static void start_write_thread(void)
{
    if (g_write_thread_running) return;
    
#ifdef _WIN32
    HANDLE hThread = CreateThread(NULL, 0, write_thread_func, NULL, 0, NULL);
    if (hThread) {
        g_write_thread_handle = hThread;
        g_write_thread_running = 1;
    } else {
        printf("[rpp_perf] Failed to create write thread\n");
    }
#else
    if (pthread_create(&g_write_thread, NULL, write_thread_func, NULL) == 0) {
        g_write_thread_running = 1;
    } else {
        printf("[rpp_perf] Failed to create write thread\n");
    }
#endif
}

/**
 * @brief One-time initialization for synchronization primitives and writer thread.
 */
static void rpp_perf_init_once(void)
{
    if (!g_mu_initialized) {
        rpp_perf_mutex_init(&g_mu);
        g_mu_initialized = 1;
    }
    if (!g_cond_initialized) {
        rpp_perf_cond_init(&g_write_cond);
        g_cond_initialized = 1;
    }
    /* Start background writer thread. */
    start_write_thread();
    /* Register process-exit cleanup hook. */
    atexit(rpp_perf_flush_on_exit);
}

/** @brief Return current timestamp in microseconds. */
static uint64_t now_us(void)
{
#ifdef _WIN32
    FILETIME ft;
    ULARGE_INTEGER uli;
    // GetSystemTimeAsFileTime(&ft); // Precision: 1ms ~ 15.6ms 
    GetSystemTimePreciseAsFileTime(&ft); // Precision: 100ns
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    return uli.QuadPart / 10;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
#endif
}

/** @brief Return current process ID. */
static uint32_t get_pid(void)
{
#ifdef _WIN32
    return (uint32_t)_getpid();
#else
    return (uint32_t)getpid();
#endif
}

/** @brief Return current thread ID in a cross-platform format. */
uint64_t rpp_perf_thread_self(void)
{
#ifdef _WIN32
    return (uint64_t)GetCurrentThreadId();
#else
    return (uint64_t)pthread_self();
#endif
}

/** @brief Return configured trace directory or platform-specific default path. */
static const char* default_trace_dir(void)
{
    if (g_trace_dir[0] != '\0') {
        return g_trace_dir;
    }
#ifdef _WIN32
    return "C:\\temp\\rpp_perf";
#else
    return "/tmp/rpp_perf";
#endif
}

/** @brief Create directory tree recursively for a target path. */
static int mkdir_recursive(const char* path)
{
    char tmp[512];
    char* p = NULL;
    size_t len;
    int result = 0;

    if (!path || path[0] == '\0') {
        return -1;
    }

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (len == 0) {
        return -1;
    }
    
    /* Normalize path separators on Windows */
#ifdef _WIN32
    for (p = tmp; *p; p++) {
        if (*p == '/') {
            *p = '\\';
        }
    }
    p = tmp;
#endif

    /* Remove trailing separator */
    if (tmp[len - 1] == PATH_SEP) {
        tmp[len - 1] = 0;
        len--;
    }

    /* Handle Windows drive letters (C:, D:, etc.) */
#ifdef _WIN32
    if (len >= 2 && tmp[1] == ':') {
        p = tmp + 3;
    } else {
        p = tmp + 1;
    }
#else
    p = tmp + 1;
#endif

    for (; *p; p++) {
        if (*p == PATH_SEP) {
            *p = 0;
#ifdef _WIN32
            if (_access(tmp, 0) != 0) {
                result = _mkdir(tmp);
                if (result != 0) {
                    int err = errno;
                    if (err != EEXIST && err != EACCES) {
                        return -1;
                    }
                }
            }
#else
            if (access(tmp, F_OK) != 0) {
                result = mkdir(tmp, 0755);
                if (result != 0) {
                    int err = errno;
                    if (err != EEXIST) {
                        return -1;
                    }
                }
            }
#endif
            *p = PATH_SEP;
        }
    }

#ifdef _WIN32
    if (_access(tmp, 0) != 0) {
        result = _mkdir(tmp);
        if (result != 0) {
            int err = errno;
            if (err != EEXIST && err != EACCES) {
                return -1;
            }
        }
    }
#else
    if (access(tmp, F_OK) != 0) {
        result = mkdir(tmp, 0755);
        if (result != 0) {
            int err = errno;
            if (err != EEXIST) {
                return -1;
            }
        }
    }
#endif

    return 0;
}

/** @brief Flush thread-local event batch into a window ring buffer (mutex held). */
static void flush_tls_locked(rpp_perf_window_t* window)
{
    if (g_tls_count == 0 || !window) return;
    
    for (uint32_t i = 0; i < g_tls_count; i++) {
        /* Use circular buffer for long-term recording */
        if (window->event_write_pos < RPP_PERF_MAX_EVENTS) {
            window->events[window->event_write_pos] = g_tls_events[i];
            window->event_write_pos++;
            if (window->event_count < RPP_PERF_MAX_EVENTS) {
                window->event_count++;
            }
        } else {
            /* Wrap around - overwrite oldest events */
            window->event_write_pos = 0;
            window->events[window->event_write_pos] = g_tls_events[i];
            window->event_write_pos++;
            if (window->event_count < RPP_PERF_MAX_EVENTS) {
                window->event_count++;
            }
        }
    }
    g_tls_count = 0;
}

/** @brief Resolve thread name by ID or generate a fallback name. */
static const char* thread_name_locked(uint64_t tid, char* buf, size_t buf_sz)
{
    /* Ensure buf is initialized to 0. */
    if (buf && buf_sz > 0) {
        buf[0] = '\0';
    }
    for (uint32_t i = 0; i < g_thread_name_count; i++) {
        if (g_thread_names[i].tid == tid) {
            /* Ensure returned string is valid and NUL-terminated. */
            if (g_thread_names[i].name && g_thread_names[i].name[0] != '\0') {
                return g_thread_names[i].name;
            }
        }
    }
    (void)snprintf(buf, buf_sz, "thread_%" PRIu64, tid);
    return buf;
}

/** @brief Write JSON-escaped text directly to file stream. */
static void escape_json_to_file(FILE* fp, const char* s)
{
    if (!fp) return;
    if (!s) {
        /* For NULL input, emit an empty string token. */
        return;
    }
    /* Safely traverse characters without reading invalid memory. */
    for (const char* p = s; *p != '\0'; p++) {
        switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default: {
                if ((unsigned char)*p < 0x20) {
                    fprintf(fp, "\\u%04X", (unsigned char)*p);
                } else {
                    fputc(*p, fp);
                }
                break;
            }
        }
    }
}

/** @brief Acquire exclusive file lock for a trace file stream. */
static int lock_file(FILE* fp)
{
#ifdef _WIN32
    int fd = _fileno(fp);
    if (fd < 0) return -1;
    HANDLE hFile = (HANDLE)_get_osfhandle(fd);
    if (hFile == INVALID_HANDLE_VALUE) return -1;
    OVERLAPPED overlapped = {0};
    if (!LockFileEx(hFile, LOCKFILE_EXCLUSIVE_LOCK, 0, 0, 0xFFFFFFFF, &overlapped)) {
        return -1;
    }
    return 0;
#else
    int fd;
    extern int fileno(FILE *);
    fd = fileno(fp);
    if (fd < 0) return -1;
    struct flock lock;
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;
    return fcntl(fd, F_SETLKW, &lock);
#endif
}

/** @brief Release previously acquired file lock for a trace file stream. */
static int unlock_file(FILE* fp)
{
#ifdef _WIN32
    int fd = _fileno(fp);
    if (fd < 0) return -1;
    HANDLE hFile = (HANDLE)_get_osfhandle(fd);
    if (hFile == INVALID_HANDLE_VALUE) return -1;
    OVERLAPPED overlapped = {0};
    if (!UnlockFileEx(hFile, 0, 0xFFFFFFFF, 0, &overlapped)) {
        return -1;
    }
    return 0;
#else
    int fd;
    extern int fileno(FILE *);
    fd = fileno(fp);
    if (fd < 0) return -1;
    struct flock lock;
    lock.l_type = F_UNLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;
    return fcntl(fd, F_SETLK, &lock);
#endif
}

/** @brief Truncate file stream to a target byte size. */
static int truncate_file(FILE* fp, long size)
{
#ifdef _WIN32
    int fd = _fileno(fp);
    if (fd < 0) return -1;
    return _chsize(fd, size);
#else
    int fd;
    extern int fileno(FILE *);
    fd = fileno(fp);
    if (fd < 0) return -1;
    return ftruncate(fd, size);
#endif
}

/** @brief Resolve thread name using a captured snapshot for async writing. */
static const char* thread_name_from_snapshot(uint64_t tid, char* buf, size_t buf_sz,
                                             const rpp_perf_thread_name_t* snapshot, uint32_t snapshot_count)
{
    /* Ensure buf is initialized to 0. */
    if (buf && buf_sz > 0) {
        buf[0] = '\0';
    }
    if (snapshot) {
        for (uint32_t i = 0; i < snapshot_count; i++) {
            if (snapshot[i].tid == tid) {
                /* Ensure returned string is valid and NUL-terminated. */
                if (snapshot[i].name && snapshot[i].name[0] != '\0') {
                    return snapshot[i].name;
                }
            }
        }
    }
    (void)snprintf(buf, buf_sz, "thread_%" PRIu64, tid);
    return buf;
}

/** @brief Build unique trace file path for a specific window. */
static void make_trace_file_path(char* path, size_t path_sz, uint32_t window_id, const char* window_name)
{
    const char* dir = default_trace_dir();
    char safe_name[64];
    (void)snprintf(safe_name, sizeof(safe_name), "%s", window_name ? window_name : "trace");
    /* Sanitize filename by replacing special characters. */
    for (char* p = safe_name; *p; p++) {
        if (*p == '/' || *p == '\\' || *p == ':' || *p == '*' || *p == '?' || 
            *p == '"' || *p == '<' || *p == '>' || *p == '|') {
            *p = '_';
        }
    }
#ifdef _WIN32
    (void)snprintf(path, path_sz, "%s\\%s_%u_%09u.trace", dir, safe_name, window_id, (unsigned)get_pid());
#else
    (void)snprintf(path, path_sz, "%s/%s_%u_%09u.trace", dir, safe_name, window_id, (unsigned)get_pid());
#endif
}

/** @brief Find trailing JSON array end bracket position in an existing trace file. */
static long find_json_array_end(FILE* fp)
{
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    if (file_size == 0) {
        return -1; /* Empty file. */
    }
    
    /* Search backward from file end and skip whitespace. */
    long pos = file_size - 1;
    long bracket_pos = -1;
    
    while (pos >= 0) {
        fseek(fp, pos, SEEK_SET);
        int c = fgetc(fp);
        if (c == EOF) {
            break;
        }
        
        if (c == ']') {
            bracket_pos = pos;
            break; /* Found closing bracket. */
        } else if (c == '\n' || c == '\r' || c == ' ' || c == '\t') {
            /* Skip whitespace and continue backward scan. */
            pos--;
            continue;
        } else {
            /* Found non-whitespace char that is not ']', possible malformed JSON tail. */
            break;
        }
    }
    
    return bracket_pos; /* Position of ']', or -1 when not found. */
}

/**
 * @brief Batch-write events to a trace file (async path, lock-free relative to global mutex).
 * @param window_id Trace window identifier.
 * @param window_name Trace window name.
 * @param events Event array to serialize.
 * @param count Number of events in @p events.
 * @param append Non-zero to append to an existing trace file.
 * @param thread_names_snapshot Snapshot of thread-name mappings.
 * @param thread_name_count_snapshot Number of snapshot entries.
 */
static void write_trace_file_batch(uint32_t window_id, const char* window_name, 
                                   const rpp_perf_event_t* events, 
                                   uint32_t count, int append,
                                   const rpp_perf_thread_name_t* thread_names_snapshot, 
                                   uint32_t thread_name_count_snapshot)
{
    if (count == 0 && !append) return;

    const char* dir = default_trace_dir();
    if (mkdir_recursive(dir) != 0) {
        printf("[rpp_perf] Failed to create directory: %s\n", dir);
        return;
    }

    char path[512];
    make_trace_file_path(path, sizeof(path), window_id, window_name);

    FILE* fp = NULL;
    int need_init_json = 0;
    
    if (append) {
        /* Try read/write open first; create file if missing. */
        fp = fopen(path, "r+");
        if (!fp) {
            /* File does not exist, create with write mode. */
            fp = fopen(path, "w");
            if (!fp) {
                printf("[rpp_perf] Failed to open file: %s\n", path);
                return;
            }
            /* Newly created file needs JSON array initialization. */
            need_init_json = 1;
            append = 0;
        }
    } else {
        fp = fopen(path, "w");
        if (!fp) {
            printf("[rpp_perf] Failed to create file: %s\n", path);
            return;
        }
        need_init_json = 1;
    }

    /* Acquire file lock. */
    if (lock_file(fp) != 0) {
        printf("[rpp_perf] Failed to acquire file lock: %s\n", path);
        fclose(fp);
        return;
    }

    /* In append mode, trim trailing ']' and add comma before new events. */
    if (append) {
        long bracket_pos = find_json_array_end(fp);
        if (bracket_pos >= 0) {
            /* Found trailing ']' (or ']\n'), truncate there and append comma. */
            truncate_file(fp, bracket_pos);
            fseek(fp, 0, SEEK_END);
            fputc(',', fp);
        } else {
            /* No valid closing bracket found; file may be empty or malformed. */
            fseek(fp, 0, SEEK_END);
            long file_size = ftell(fp);
            if (file_size == 0) {
                /* Empty file: initialize JSON structure. */
                need_init_json = 1;
            } else {
                /* Existing but malformed file: attempt a conservative repair with comma handling. */
                fseek(fp, 0, SEEK_END);
                /* Check last character and append comma when needed. */
                if (file_size > 0) {
                    fseek(fp, -1, SEEK_END);
                    int last_char = fgetc(fp);
                    if (last_char != ',' && last_char != '[') {
                        fseek(fp, 0, SEEK_END);
                        fputc(',', fp);
                    } else if (last_char == '[') {
                        /* File starts with '[' and has no data: valid initial state. */
                        fseek(fp, 0, SEEK_END);
                    } else {
                        /* Already ends with comma, no extra comma needed. */
                        fseek(fp, 0, SEEK_END);
                    }
                }
            }
        }
    }
    
    /* Write JSON prefix when initialization is required. */
    if (need_init_json) {
        fputs("[", fp);
        fprintf(fp, "{\"name\":\"TraceStart\",\"ph\":\"i\",\"ts\":%" PRIu64 ",\"pid\":0}", now_us());
        if (count > 0) {
            fputc(',', fp);
        }
    }

    /* Stream events directly to file to avoid large intermediate buffers. */
    for (uint32_t i = 0; i < count; i++) {
        const rpp_perf_event_t* e = &events[i];
        char tid_buf[32];
        const char* tid_name = thread_name_from_snapshot(e->tid, tid_buf, sizeof(tid_buf),
                                                         thread_names_snapshot, thread_name_count_snapshot);

        fputc('{', fp);
        fputs("\"name\":\"", fp);
        escape_json_to_file(fp, e->name);
        fputs("\",\"cat\":\"", fp);
        escape_json_to_file(fp, e->cat);
        fprintf(fp, "\",\"ph\":\"%c\",\"ts\":%" PRIu64 ",\"pid\":%u,\"tid\":\"", 
                (char)e->ph, e->ts_us, e->pid);
        escape_json_to_file(fp, tid_name ? tid_name : "");
        fputc('"', fp);
        
        if (e->ph == RPP_PERF_EVT_C) {
            fprintf(fp, ",\"args\":{\"value\":%" PRIu64 "}", e->value);
        }
        
        fputc('}', fp);
        if (i < count - 1) {
            fputc(',', fp);
        }
    }

    /* Write JSON array suffix. */
    fputs("]\n", fp);
    
    /* Ensure data is flushed to disk. */
    fflush(fp);
    
    printf("[rpp_perf] Trace file written: %s (event count: %u)\n", path, count);
    unlock_file(fp);
    fclose(fp);
}

/**
 * @brief Write all buffered events for a window synchronously.
 * @param window_id Trace window identifier.
 * @param window_name Trace window name.
 */
static void write_trace_file_locked(uint32_t window_id, const char* window_name)
{
    rpp_perf_window_t* window = find_window_locked(window_id);
    if (!window || window->event_count == 0) return;

    const char* dir = default_trace_dir();
    if (mkdir_recursive(dir) != 0) {
        printf("[rpp_perf] Failed to create directory: %s\n", dir);
        window->event_count = 0;
        window->event_write_pos = 0;
        return;
    }

    char path[512];
    make_trace_file_path(path, sizeof(path), window_id, window_name);

    FILE* fp = fopen(path, "r+");
    int need_init_json = 0;
    
    if (!fp) {
        /* File does not exist, create a new one. */
        fp = fopen(path, "w");
        if (!fp) {
            printf("[rpp_perf] Failed to create file: %s\n", path);
            window->event_count = 0;
            window->event_write_pos = 0;
            return;
        }
        need_init_json = 1;
    }

    /* Acquire file lock. */
    if (lock_file(fp) != 0) {
        printf("[rpp_perf] Failed to acquire file lock: %s\n", path);
        fclose(fp);
        window->event_count = 0;
        window->event_write_pos = 0;
        return;
    }

    /* Check whether file already contains JSON content. */
    if (!need_init_json) {
        long bracket_pos = find_json_array_end(fp);
        if (bracket_pos >= 0) {
            /* Found trailing ']' (or ']\n'), truncate and append comma. */
            truncate_file(fp, bracket_pos);
            fseek(fp, 0, SEEK_END);
            fputc(',', fp);
        } else {
            /* No valid array ending found; handle empty/malformed cases. */
            fseek(fp, 0, SEEK_END);
            long file_size = ftell(fp);
            if (file_size == 0) {
                /* Empty file: initialize JSON structure. */
                need_init_json = 1;
            } else {
                /* Existing but malformed file: attempt conservative comma repair. */
                fseek(fp, 0, SEEK_END);
                /* Append comma when last character is not a comma. */
                if (file_size > 0) {
                    fseek(fp, -1, SEEK_END);
                    int last_char = fgetc(fp);
                    if (last_char != ',' && last_char != '[') {
                        fseek(fp, 0, SEEK_END);
                        fputc(',', fp);
                    } else if (last_char == '[') {
                        /* File starts with '[' and has no data: valid initial state. */
                        fseek(fp, 0, SEEK_END);
                    } else {
                        /* Already a comma, no action needed. */
                        fseek(fp, 0, SEEK_END);
                    }
                }
            }
        }
    }
    
    /* Write JSON prefix when initialization is required. */
    if (need_init_json) {
        fputs("[", fp);
        fprintf(fp, "{\"name\":\"TraceStart\",\"ph\":\"i\",\"ts\":%" PRIu64 ",\"pid\":0}", now_us());
        if (window->event_count > 0) {
            fputc(',', fp);
        }
    }

    /* Write events in order (handle circular buffer) */
    uint32_t start_pos = (window->event_count == RPP_PERF_MAX_EVENTS) ? 
                         window->event_write_pos : 0;
    uint32_t count = window->event_count;
    
    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = (start_pos + i) % RPP_PERF_MAX_EVENTS;
        const rpp_perf_event_t* e = &window->events[idx];
        char tid_buf[32];
        const char* tid_name = thread_name_locked(e->tid, tid_buf, sizeof(tid_buf));

        fputc('{', fp);
        fputs("\"name\":\"", fp);
        escape_json_to_file(fp, e->name);
        fputs("\",\"cat\":\"", fp);
        escape_json_to_file(fp, e->cat);
        fprintf(fp, "\",\"ph\":\"%c\",\"ts\":%" PRIu64 ",\"pid\":%u,\"tid\":\"", 
                (char)e->ph, e->ts_us, e->pid);
        escape_json_to_file(fp, tid_name ? tid_name : "");
        fputc('"', fp);
        
        if (e->ph == RPP_PERF_EVT_C) {
            fprintf(fp, ",\"args\":{\"value\":%" PRIu64 "}", e->value);
        }
        
        fputc('}', fp);
        if (i < count - 1) {
            fputc(',', fp);
        }
    }
    
    /* Write JSON array suffix. */
    fputs("]\n", fp);
    
    /* Ensure data is flushed to disk. */
    fflush(fp);
    
    unlock_file(fp);
    fclose(fp);
    
    printf("[rpp_perf] Trace file written: %s (event count: %u)\n", path, window->event_count);
    window->event_count = 0;
    window->event_write_pos = 0;
}

/**
 * @brief Mark a window for asynchronous flush when threshold conditions are met.
 * @param window Target window (mutex must be held).
 */
static void maybe_flush_window_locked(rpp_perf_window_t* window)
{
    if (!window || window->remaining == 0) return;
    if (window->event_count < RPP_PERF_FLUSH_THRESHOLD) return;
    /* Trigger async flush at threshold to avoid buffer saturation. */
    window->pending_write = 1;
    rpp_perf_cond_signal(&g_write_cond);
}

/**
 * @brief Push one event into the thread-local buffer and schedule flush if needed.
 * @param e Event to enqueue.
 */
static void push_event(rpp_perf_event_t e)
{
    rpp_perf_once(&g_once, rpp_perf_init_once);

    rpp_perf_mutex_lock(&g_mu);
    rpp_perf_window_t* window = get_active_window_locked();
    if (!window || window->remaining == 0) {
        rpp_perf_mutex_unlock(&g_mu);
        return;
    }
    rpp_perf_mutex_unlock(&g_mu);

    g_tls_events[g_tls_count++] = e;
    if (g_tls_count < RPP_PERF_TLS_BATCH) return;

    rpp_perf_mutex_lock(&g_mu);
    window = get_active_window_locked();
    if (window && window->remaining > 0) {
        flush_tls_locked(window);
        if (window->event_count > 0) {
            window->pending_write = 1;
        }
        maybe_flush_window_locked(window);
    }
    rpp_perf_mutex_unlock(&g_mu);
}

/**
 * @brief Create a trace window and bind it as current thread active window.
 * @param name Optional trace window name.
 * @param event_count Number of expected scope-end events before final flush.
 * @return Window ID on success, or 0 on failure.
 */
uint32_t rpp_perf_add_trace_window(const char* name, uint32_t event_count)
{
#ifndef ENABLE_PERFETTO_TRACE
    return 0;
#endif
    rpp_perf_once(&g_once, rpp_perf_init_once);
    rpp_perf_mutex_lock(&g_mu);
    
    /* Find a free window slot and initialize it. */
    rpp_perf_window_t* window = NULL;
    for (uint32_t i = 0; i < RPP_PERF_MAX_WINDOWS; i++) {
        if (!g_windows[i].active) {
            window = &g_windows[i];
            window->window_id = i + 1;
            break;
        }
    }
    
    if (!window) {
        printf("[rpp_perf] Maximum window limit (%u) reached, cannot create new window\n", RPP_PERF_MAX_WINDOWS);
        rpp_perf_mutex_unlock(&g_mu);
        return 0; /* Failure: reached max window count. */
    }
    
    /* Initialize window state. */
    (void)snprintf(window->name, sizeof(window->name), "%s", name ? name : "trace");
    window->remaining = event_count;
    window->event_count = 0;
    window->event_write_pos = 0;
    window->pending_write = 0;
    window->active = 1;
    g_window_mask |= (0x01 << (window->window_id - 1));
    
    /* Set newly created window as current thread active window. */
    g_tls_active_window_id = window->window_id;
    
    uint32_t window_id = window->window_id;
    printf("[rpp_perf] Trace window: %s (window id: %u,event count: %u)\n", window->name, window->window_id, window->remaining);
    rpp_perf_mutex_unlock(&g_mu);
    return window_id;
}

/**
 * @brief Remove a trace window and flush its remaining buffered events.
 * @param window_id Window ID to remove.
 * @return Removed window ID on success, or 0 on failure.
 */
uint32_t rpp_perf_remove_trace_window(uint32_t window_id)
{
#ifndef ENABLE_PERFETTO_TRACE
    return 0;
#endif
    if (window_id > RPP_PERF_MAX_WINDOWS)
    {
        printf("[rpp_perf] Invalide window id (%u) occured, cannot remove window\n", window_id);
    }
    
    rpp_perf_once(&g_once, rpp_perf_init_once);
    rpp_perf_mutex_lock(&g_mu);
    
    /* Find target window to remove. */
    rpp_perf_window_t* window = find_window_locked(window_id);
    if (!window) {
        printf("[rpp_perf] Failed to remove trace window: window id %u does not exist or is not active\n", window_id);
        rpp_perf_mutex_unlock(&g_mu);
        return 0;
    }
    
    /* Clear TLS active marker if it points to the removed window. */
    if (g_tls_active_window_id == window_id) {
        g_tls_active_window_id = 0;
    }
    
    /* Flush current thread TLS buffer to this window. */
    if (g_tls_count > 0) {
        flush_tls_locked(window);
    }
    
    // /* If window still has pending events, request flush. */
    // if (window->event_count > 0) {
    //     window->pending_write = 1;
    //     rpp_perf_cond_signal(&g_write_cond);
    // }
    
    /* Mark window inactive. */
    write_trace_file_locked(window->window_id, window->name);
    window->active = 0;
    g_window_mask &= ~(0x01 << (window->window_id - 1));
    printf("[rpp_perf] Trace window removed: %s (window id: %u)\n", window->name, window_id);
    rpp_perf_mutex_unlock(&g_mu);
    return window_id;
}

/**
 * @brief Set active trace window for current thread.
 * @param window_id Target window ID.
 */
void rpp_perf_set_active_window(uint32_t window_id)
{
#ifndef ENABLE_PERFETTO_TRACE
    return;
#endif
    rpp_perf_once(&g_once, rpp_perf_init_once);
    rpp_perf_mutex_lock(&g_mu);
    rpp_perf_window_t* window = find_window_locked(window_id);
    if (window && window->active) {
        g_tls_active_window_id = window_id;
    } else {
        printf("[rpp_perf] Failed to set active window: window ID %u does not exist or is not active\n", window_id);
    }
    rpp_perf_mutex_unlock(&g_mu);
}

/**
 * @brief Set output directory for generated trace files.
 * @param dir Target directory path.
 */
void rpp_perf_set_trace_dir(const char* dir)
{
#ifndef ENABLE_PERFETTO_TRACE
    return;
#endif
    if (!dir) return;
    rpp_perf_once(&g_once, rpp_perf_init_once);
    rpp_perf_mutex_lock(&g_mu);
    (void)snprintf(g_trace_dir, sizeof(g_trace_dir), "%s", dir);
    rpp_perf_mutex_unlock(&g_mu);
}

/**
 * @brief Register or update a human-readable thread name.
 * @param thread_id Thread ID returned by rpp_perf_thread_self().
 * @param name Display name written into trace events.
 */
void rpp_perf_set_thread_name(uint64_t thread_id, const char* name)
{
#ifndef ENABLE_PERFETTO_TRACE
    return;
#endif
    if (!thread_id || !name) return;
    rpp_perf_once(&g_once, rpp_perf_init_once);

    rpp_perf_mutex_lock(&g_mu);
    for (uint32_t i = 0; i < g_thread_name_count; i++) {
        if (g_thread_names[i].tid == thread_id) {
            (void)snprintf(g_thread_names[i].name, sizeof(g_thread_names[i].name), "%s", name);
            rpp_perf_mutex_unlock(&g_mu);
            return;
        }
    }
    if (g_thread_name_count < RPP_PERF_MAX_THREAD_NAMES) {
        g_thread_names[g_thread_name_count].tid = thread_id;
        (void)snprintf(g_thread_names[g_thread_name_count].name, sizeof(g_thread_names[g_thread_name_count].name), "%s", name);
        g_thread_name_count++;
    } else {
        printf("[rpp_perf] Maximum thread name limit (%u) reached, cannot add new thread name\n", RPP_PERF_MAX_THREAD_NAMES);
    }
    rpp_perf_mutex_unlock(&g_mu);
}

/**
 * @brief Create compact category text from source file and line.
 * @param out Output buffer.
 * @param out_sz Output buffer size.
 * @param file Source file path.
 * @param line Source line number.
 */
static void cat_buf(char* out, size_t out_sz, const char* file, int line)
{
    const char* filename = file ? file : "?";
    /* Extract filename from path (handle both '/' and '\' separators) */
    const char* last_sep = strrchr(filename, '/');
    if (!last_sep) {
        last_sep = strrchr(filename, '\\');
    }
    if (last_sep) {
        filename = last_sep + 1;
    }
    (void)snprintf(out, out_sz, "%s:%d", filename, line);
}

/**
 * @brief Start function-scope tracing using function name.
 * @param func Function name.
 * @param file Source file path.
 * @param line Source line number.
 * @return Initialized trace scope handle.
 */
rpp_perf_scope_t rpp_perf_trace_function(const char* func, const char* file, int line)
{
    return rpp_perf_trace_scope(func, file, line);
}

/**
 * @brief Start named trace scope.
 * @param name Scope name.
 * @param file Source file path.
 * @param line Source line number.
 * @return Initialized trace scope handle.
 */
rpp_perf_scope_t rpp_perf_trace_scope(const char* name, const char* file, int line)
{
    rpp_perf_scope_t s;
#ifndef ENABLE_PERFETTO_TRACE
    return s;
#endif
    snprintf(s.name, sizeof(s.name), "%s", name ? name : "");
    cat_buf(s.cat, sizeof(s.cat), file, line);
    s.pid = get_pid();
    s.tid = rpp_perf_thread_self();
    s.start_ts_us = now_us();
    s.active = 1;

    rpp_perf_event_t e = {0};
    snprintf(e.name, sizeof(e.name), "%s", s.name);
    snprintf(e.cat, sizeof(e.cat), "%s", s.cat);
    e.ph = RPP_PERF_EVT_B;
    e.ts_us = s.start_ts_us;
    e.pid = s.pid;
    e.tid = s.tid;
    e.value = 0;
    push_event(e);
    return s;
}

/**
 * @brief Record a counter event.
 * @param name Counter name.
 * @param value Counter value.
 * @param file Source file path.
 * @param line Source line number.
 */
void rpp_perf_trace_value_u64(const char* name, uint64_t value, const char* file, int line)
{
#ifndef ENABLE_PERFETTO_TRACE
    return;
#endif
    rpp_perf_event_t e = {0};
    snprintf(e.name, sizeof(e.name), "%s", name ? name : "");
    cat_buf(e.cat, sizeof(e.cat), file, line);
    e.ph = RPP_PERF_EVT_C;
    e.ts_us = now_us();
    e.pid = get_pid();
    e.tid = rpp_perf_thread_self();
    e.value = value;
    push_event(e);
}

/**
 * @brief End a previously started trace scope.
 * @param scope Scope handle created by rpp_perf_trace_scope().
 */
void rpp_perf_scope_end(rpp_perf_scope_t* scope)
{
#ifndef ENABLE_PERFETTO_TRACE
    return;
#endif
    if (!scope || !scope->active) return;
    scope->active = 0;

    rpp_perf_event_t e = {0};
    snprintf(e.name, sizeof(e.name), "%s", scope->name);
    snprintf(e.cat, sizeof(e.cat), "%s", scope->cat);
    e.ph = RPP_PERF_EVT_E;
    e.ts_us = now_us();
    e.pid = scope->pid;
    e.tid = scope->tid;
    e.value = 0;
    push_event(e);

    /* Decrement remaining count and trigger flush when window reaches zero. */
    rpp_perf_once(&g_once, rpp_perf_init_once);
    rpp_perf_mutex_lock(&g_mu);
    rpp_perf_window_t* window = get_active_window_locked();
    if (window && window->remaining > 0) {
        flush_tls_locked(window);
        window->remaining--;
        if (window->remaining == 0) {
            /* Window completed, request final flush. */
            if (window->event_count > 0) {
                window->pending_write = 1;
                rpp_perf_cond_signal(&g_write_cond);
            }
        } else if (window->event_count > 0) {
            window->pending_write = 1;
            rpp_perf_cond_signal(&g_write_cond);
        }
    }
    rpp_perf_mutex_unlock(&g_mu);
}

/**
 * @brief Flush pending trace data during process shutdown.
 */
void rpp_perf_flush_on_exit(void)
{
#ifndef ENABLE_PERFETTO_TRACE
    return;
#endif
    if (!g_mu_initialized) return;
    
    rpp_perf_mutex_lock(&g_mu);
    
    /* Mark shutdown and stop accepting new events. */
    g_shutdown = 1;
    
    /* Flush current thread TLS buffer into its active window. */
    if (g_tls_count > 0) {
        rpp_perf_window_t* window = get_active_window_locked();
        if (window && window->active) {
            /* Flush TLS even when remaining==0 so no buffered events are lost. */
            flush_tls_locked(window);
            if (window->event_count > 0) {
                window->pending_write = 1;
            }
        }
    }
    
    /* Mark all active windows for flushing remaining data. */
    for (uint32_t i = 0; i < RPP_PERF_MAX_WINDOWS; i++) {
        if (g_windows[i].active) {
            if (g_windows[i].event_count > 0) {
                g_windows[i].pending_write = 1;
            }
        }
    }
    
    /* Wake writer thread for pending windows. */
    rpp_perf_cond_signal(&g_write_cond);
    
    rpp_perf_mutex_unlock(&g_mu);
    
    /* Wait for background writer to complete. */
    if (g_write_thread_running) {
#ifdef _WIN32
        if (g_write_thread_handle != NULL) {
            /* Wait up to 5 seconds for writer thread shutdown. */
            DWORD wait_result = WaitForSingleObject(g_write_thread_handle, 5000);
            if (wait_result == WAIT_OBJECT_0) {
                CloseHandle(g_write_thread_handle);
                g_write_thread_handle = NULL;
                g_write_thread_running = 0;
            } else if (wait_result == WAIT_TIMEOUT) {
                printf("[rpp_perf] Wait for write thread completion timeout (5s)\n");
                /* On timeout, force close handle to avoid resource leak. */
                CloseHandle(g_write_thread_handle);
                g_write_thread_handle = NULL;
                g_write_thread_running = 0;
            }
        }
#else
        if (g_write_thread != 0) {
            /* Wait for thread completion. It should exit soon after shutdown signal. */
            void* thread_result = NULL;
            int result = pthread_join(g_write_thread, &thread_result);
            if (result == 0) {
                g_write_thread = 0;
                g_write_thread_running = 0;
            } else {
                printf("[rpp_perf] Failed to wait for write thread completion: %d\n", result);
                /* If join fails, try detach to avoid leaked thread resources. */
                if (result == ESRCH || result == EINVAL) {
                    pthread_detach(g_write_thread);
                    g_write_thread = 0;
                    g_write_thread_running = 0;
                }
            }
        }
#endif
    }
    
    /* Final fallback: synchronously flush any remaining active-window data. */
    rpp_perf_mutex_lock(&g_mu);
    for (uint32_t i = 0; i < RPP_PERF_MAX_WINDOWS; i++) {
        if (g_windows[i].active && g_windows[i].event_count > 0) {
            uint32_t window_id = g_windows[i].window_id;
            const char* window_name = g_windows[i].name;
            rpp_perf_mutex_unlock(&g_mu);
            write_trace_file_locked(window_id, window_name);
            rpp_perf_mutex_lock(&g_mu);
        }
    }
    rpp_perf_mutex_unlock(&g_mu);
}
