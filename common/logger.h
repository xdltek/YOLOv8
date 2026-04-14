/** @file logger.h
 *
 * @brief
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
    extern Logger gLogger;
    extern LogStreamConsumer gLogVerbose;
    extern LogStreamConsumer gLogInfo;
    extern LogStreamConsumer gLogWarning;
    extern LogStreamConsumer gLogError;
    extern LogStreamConsumer gLogFatal;

    void setReportableSeverity(Logger::Severity severity);

    Logger::Severity get_current_log_level();
    void set_log_path(const std::string& log_file_path);

    void user_visible_log(LogStreamConsumer& rt_logger, const std::string& log_path, const std::string& log_text);
    void user_visible_log(const std::string& log_text);

    template <typename... Args>
    void user_visible_stream_log(const Args&... args) {
        std::ostringstream oss;
        (oss << ... << args);
        user_visible_log(oss.str());
    }
} // namespace sample

#endif // LOGGER_H

