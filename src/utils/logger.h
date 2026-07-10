/** @file logger.h
 *
 * @brief Shared logger declarations used by parser, runtime, and demos.
 * @author XDLTek
 * COPYRIGHT(c) 2020-2022 XDLTek.
 * ALL RIGHTS RESERVED
 *
 * This is Unpublished Proprietary Source Code of XDLTek
 */
#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include "logging.h"

namespace sample
{
    //! Global loggers matching the OpenRT sample severity levels.
    extern Logger gLogger;
    extern LogStreamConsumer gLogVerbose;
    extern LogStreamConsumer gLogInfo;
    extern LogStreamConsumer gLogWarning;
    extern LogStreamConsumer gLogError;
    extern LogStreamConsumer gLogFatal;

    //! Configure the reportable severity for every shared log stream.
    void setReportableSeverity(Logger::Severity severity);

    //! Return the current global logger severity.
    Logger::Severity get_current_log_level();

    //! Set the optional file path used by the logging backend.
    void set_log_path(const std::string& log_file_path);

    //! Emit a user-visible message through a selected stream and optional file.
    void user_visible_log(LogStreamConsumer& rt_logger, const std::string& log_path, const std::string& log_text);

    //! Emit a user-visible info message to the console.
    void user_visible_log(const std::string& log_text);

    //! Build a message from streamable arguments and emit it as a user-visible info log.
    template <typename... Args>
    void user_visible_stream_log(const Args&... args) {
        std::ostringstream oss;
        (oss << ... << args);
        user_visible_log(oss.str());
    }
} // namespace sample

#endif // LOGGER_H
