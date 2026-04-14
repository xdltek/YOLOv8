/** @file logger.cpp
 *
 * @brief
 * @author XDLTek
 * COPYRIGHT(c) 2020-2022 XDLTek.
 * ALL RIGHTS RESERVED
 *
 * This is Unpublished Proprietary Source Code of XDLTek
 */

#include "logger.h"
#include "ErrorRecorder.h"
#include "logging.h"

SampleErrorRecorder gRecorder;
namespace sample
{
    Logger gLogger{Logger::Severity::kERROR};
    LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
    LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
    LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
    LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
    LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};

    void setReportableSeverity(Logger::Severity severity)
    {
        gLogger.setReportableSeverity(severity);
        gLogVerbose.setReportableSeverity(severity);
        gLogInfo.setReportableSeverity(severity);
        gLogWarning.setReportableSeverity(severity);
        gLogError.setReportableSeverity(severity);
        gLogFatal.setReportableSeverity(severity);
    }

    Logger::Severity get_current_log_level()
    {
        return  gLogger.getReportableSeverity();
    }

    void set_log_path(const std::string& log_file_path)
    {
        sample::set_log_file_path_111(log_file_path);
    }

    void user_visible_log(LogStreamConsumer& rt_logger, const std::string& log_path, const std::string& log_text)
    {
        rt_logger << log_text << std::endl;
        std::cout << log_text << std::endl;

//        if (get_current_log_level() == Logger::Severity::kVERBOSE)
//        {
//            return;
//        }


        if (log_path.empty())
        {
            return;
        }

        std::ofstream file_stream;
        file_stream.open(log_path, std::ios_base::app | std::ios_base::in);
        if (!file_stream.is_open()) {
            throw std::runtime_error("cannot open log file, path: " + log_path);
        }

        file_stream << log_text << std::endl;
        file_stream.close();
    }

    void user_visible_log(const std::string& log_text)
    {
        return user_visible_log(gLogInfo, "", log_text);
    }
} // namespace sample
