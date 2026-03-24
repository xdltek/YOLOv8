/** @file logging.h
 *
 * @brief Logging backend for runtime severity logs and sample test status messages.
 */
#ifndef RPPRT_LOGGING_H
#define RPPRT_LOGGING_H

#include <ctime>
#include <iostream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <fstream>
#include <string>

#include "InferRuntimeCommon.h"

namespace sample
{
    using Severity = infer1::ILogger::Severity;

    const std::string& get_log_file_path();
    void set_log_file_path_111(const std::string& path);

    class Logger;
    extern Logger gLogger;

    class LogStreamConsumerBuffer : public std::stringbuf
    {
    public:
        LogStreamConsumerBuffer(std::ostream& stream, const std::string& prefix, bool shouldLog)
                : mOutput(stream)
                , mPrefix(prefix)
                , mShouldLog(shouldLog)
        {
        }

        LogStreamConsumerBuffer(LogStreamConsumerBuffer&& other)
                : mOutput(other.mOutput)
        {
        }

        ~LogStreamConsumerBuffer()
        {
            // std::streambuf::pbase() gives a pointer to the beginning of the buffered part of the output sequence
            // std::streambuf::pptr() gives a pointer to the current position of the output sequence
            // if the pointer to the beginning is not equal to the pointer to the current position,
            // call putOutput() to log the output to the stream
            if (pbase() != pptr())
            {
                putOutput();
            }
        }

        // synchronizes the stream buffer and returns 0 on success
        // synchronizing the stream buffer consists of inserting the buffer contents into the stream,
        // resetting the buffer and flushing the stream
        virtual int sync()
        {
            putOutput();
            return 0;
        }

        void putOutput()
        {
            if (mShouldLog)
            {
                std::string log_text = str();

                // prepend timestamp
                std::time_t timestamp = std::time(nullptr);
                tm* tm_local = std::localtime(&timestamp);
                mOutput << "[";
                mOutput << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
                mOutput << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
                mOutput << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
                mOutput << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
                mOutput << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
                mOutput << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
                // std::stringbuf::str() gets the string contents of the buffer
                // insert the buffer contents pre-appended by the appropriate prefix into the stream
                mOutput << mPrefix << log_text;

                const std::string& log_file = get_log_file_path();
                if (!log_file.empty())
                {
                    std::ofstream file_stream;
                    file_stream.open(log_file, std::ios_base::app | std::ios_base::in);
                    if (!file_stream.is_open()) {
                        throw std::runtime_error("cannot open log file");
                    }

                    file_stream << mPrefix << log_text;

                    file_stream.close();
                }

                // set the buffer to empty
                str("");
                // flush the stream
                mOutput.flush();
            }
        }

        void setShouldLog(bool shouldLog)
        {
            mShouldLog = shouldLog;
        }

    private:
        std::ostream& mOutput;
        std::string mPrefix;
        bool mShouldLog;
    };

    //!
    //! \class LogStreamConsumerBase
    //! \brief Convenience object used to initialize LogStreamConsumerBuffer before std::ostream in LogStreamConsumer
    //!
    class LogStreamConsumerBase
    {
    public:
        LogStreamConsumerBase(std::ostream& stream, const std::string& prefix, bool shouldLog)
                : mBuffer(stream, prefix, shouldLog)
        {
        }

    protected:
        LogStreamConsumerBuffer mBuffer;
    };

    //!
    //! \class LogStreamConsumer
    //! \brief Convenience object used to facilitate use of C++ stream syntax when logging messages.
    //!  Order of base classes is LogStreamConsumerBase and then std::ostream.
    //!  This is because the LogStreamConsumerBase class is used to initialize the LogStreamConsumerBuffer member field
    //!  in LogStreamConsumer and then the address of the buffer is passed to std::ostream.
    //!  This is necessary to prevent the address of an uninitialized buffer from being passed to std::ostream.
    //!  Please do not change the order of the parent classes.
    //!
    class LogStreamConsumer : protected LogStreamConsumerBase, public std::ostream
    {
    public:
        //! \brief Creates a LogStreamConsumer which logs messages with level severity.
        //!  Reportable severity determines if the messages are severe enough to be logged.
        LogStreamConsumer(Severity reportableSeverity, Severity severity)
                : LogStreamConsumerBase(severityOstream(severity), severityPrefix(severity), severity <= reportableSeverity)
                , std::ostream(&mBuffer) // links the stream buffer with the stream
                , mShouldLog(severity <= reportableSeverity)
                , mSeverity(severity)
        {
        }

        LogStreamConsumer(LogStreamConsumer&& other)
                : LogStreamConsumerBase(severityOstream(other.mSeverity), severityPrefix(other.mSeverity), other.mShouldLog)
                , std::ostream(&mBuffer) // links the stream buffer with the stream
                , mShouldLog(other.mShouldLog)
                , mSeverity(other.mSeverity)
        {
        }

        void setReportableSeverity(Severity reportableSeverity)
        {
            mShouldLog = mSeverity <= reportableSeverity;
            mBuffer.setShouldLog(mShouldLog);
        }

    private:
        static std::ostream& severityOstream(Severity severity)
        {
            return severity >= Severity::kINFO ? std::cout : std::cerr;
        }

        static std::string severityPrefix(Severity severity)
        {
            // return "";

            switch (severity)
            {
                case Severity::kINTERNAL_ERROR: return "[F] ";
                case Severity::kERROR: return "[E] ";
                case Severity::kWARNING: return "[W] ";
                case Severity::kINFO: return "[I] ";
                case Severity::kVERBOSE: return "[V] ";
                default: return "";
            }
        }

        bool mShouldLog;
        Severity mSeverity;
    };

    //! \class Logger
    //!
    //! \brief Class which manages logging of TensorRT tools and samples
    //!
    //! \details This class provides a common interface for TensorRT tools and samples to log information to the console, and
    //! supports logging two types of messages:
    //!
    //! - Debugging messages with an associated severity (info, warning, error, or internal error/fatal)
    //! - Test pass/fail messages
    //!
    //! The advantage of having all samples use this class for logging as opposed to emitting directly to stdout/stderr is that
    //! the logic for controlling the verbosity and formatting of sample output is centralized in one location.
    //!
    //! In the future, this class could be extended to support dumping test results to a file in some standard format
    //! (for example, JUnit XML), and providing additional metadata (e.g. timing the duration of a test run).
    //!
    //! TODO: For backwards compatibility with existing samples, this class inherits infer1::ILogger directly.
    //! This makes it harder to cleanly separate runtime-library messages from sample-application messages.
    //!
    //! In the future (once all samples are updated to use Logger::getTRTLogger() to access the ILogger) we can refactor the class
    //! to eliminate the inheritance and instead make the infer1::ILogger implementation a member of the Logger object.

    class Logger : public infer1::ILogger
    {
    public:
        Logger(Severity severity = Severity::kVERBOSE)
                : mReportableSeverity(severity)
        {
        }

        //!
        //! \enum TestResult
        //! \brief Represents the state of a given test
        //!
        enum class TestResult
        {
            kRUNNING, //!< The test is running
            kPASSED,  //!< The test passed
            kFAILED,  //!< The test failed
            kWAIVED   //!< The test was waived
        };

        //!
        //! \brief Forward-compatible method for retrieving the nvinfer::ILogger associated with this Logger
        //! \return The infer1::ILogger associated with this Logger
        //!
        //! TODO Once all samples are updated to use this method to register the logger with TensorRT,
        //! we can eliminate the inheritance of Logger from ILogger
        //!
        infer1::ILogger& getLogger()
        {
            return *this;
        }

        //!
        //! \brief Implementation of the infer1::ILogger::log() virtual method
        //!
        //! Note samples should not be calling this function directly; it will eventually go away once we eliminate the inheritance from
        //! infer1::ILogger
        //!
        void log(Severity severity, const char* msg) override
        {
            LogStreamConsumer(mReportableSeverity, severity) << "[OpenRT] " << std::string(msg) << std::endl;
        }

        //!
        //! \brief Method for controlling the verbosity of logging output
        //!
        //! \param severity The logger will only emit messages that have severity of this level or higher.
        //!
        void setReportableSeverity(Severity severity)
        {
            mReportableSeverity = severity;
        }

        //!
        //! \brief Opaque handle that holds logging information for a particular test
        //!
        //! This object is an opaque handle to information used by the Logger to print test results.
        //! The sample must call Logger::defineTest() in order to obtain a TestAtom that can be used
        //! with Logger::reportTest{Start,End}().
        //!
        class TestAtom
        {
        public:
            TestAtom(TestAtom&&) = default;

        private:
            friend class Logger;

            TestAtom(bool started, const std::string& name, const std::string& cmdline)
                    : mStarted(started)
                    , mName(name)
                    , mCmdline(cmdline)
            {
            }

            bool mStarted;
            std::string mName;
            std::string mCmdline;
        };

        //!
        //! \brief Define a test for logging
        //!
        //! \param[in] name The name of the test.  This should be a string starting with
        //!                  "TensorRT" and containing dot-separated strings containing
        //!                  the characters [A-Za-z0-9_].
        //!                  For example, "TensorRT.sample_googlenet"
        //! \param[in] cmdline The command line used to reproduce the test
        //
        //! \return a TestAtom that can be used in Logger::reportTest{Start,End}().
        //!
        static TestAtom defineTest(const std::string& name, const std::string& cmdline)
        {
            return TestAtom(false, name, cmdline);
        }

        //!
        //! \brief A convenience overloaded version of defineTest() that accepts an array of command-line arguments
        //!        as input
        //!
        //! \param[in] name The name of the test
        //! \param[in] argc The number of command-line arguments
        //! \param[in] argv The array of command-line arguments (given as C strings)
        //!
        //! \return a TestAtom that can be used in Logger::reportTest{Start,End}().
        static TestAtom defineTest(const std::string& name, int argc, char const* const* argv)
        {
            auto cmdline = genCmdlineString(argc, argv);
            return defineTest(name, cmdline);
        }

        //!
        //! \brief Report that a test has started.
        //!
        //! \pre reportTestStart() has not been called yet for the given testAtom
        //!
        //! \param[in] testAtom The handle to the test that has started
        //!
        static void reportTestStart(TestAtom& testAtom)
        {
            reportTestResult(testAtom, TestResult::kRUNNING);
            testAtom.mStarted = true;
        }

        //!
        //! \brief Report that a test has ended.
        //!
        //! \pre reportTestStart() has been called for the given testAtom
        //!
        //! \param[in] testAtom The handle to the test that has ended
        //! \param[in] result The result of the test. Should be one of TestResult::kPASSED,
        //!                   TestResult::kFAILED, TestResult::kWAIVED
        //!
        static void reportTestEnd(const TestAtom& testAtom, TestResult result)
        {
            reportTestResult(testAtom, result);
        }

        static int reportPass(const TestAtom& testAtom)
        {
            reportTestEnd(testAtom, TestResult::kPASSED);
            return EXIT_SUCCESS;
        }

        static int reportFail(const TestAtom& testAtom)
        {
            reportTestEnd(testAtom, TestResult::kFAILED);
            return EXIT_FAILURE;
        }

        static int reportWaive(const TestAtom& testAtom)
        {
            reportTestEnd(testAtom, TestResult::kWAIVED);
            return EXIT_SUCCESS;
        }

        static int reportTest(const TestAtom& testAtom, bool pass)
        {
            return pass ? reportPass(testAtom) : reportFail(testAtom);
        }

        Severity getReportableSeverity() const
        {
            return mReportableSeverity;
        }

    private:
        //!
        //! \brief returns an appropriate string for prefixing a log message with the given severity
        //!
        static const char* severityPrefix(Severity severity)
        {
            switch (severity)
            {
                case Severity::kINTERNAL_ERROR: return "[F] ";
                case Severity::kERROR: return "[E] ";
                case Severity::kWARNING: return "[W] ";
                case Severity::kINFO: return "[I] ";
                case Severity::kVERBOSE: return "[V] ";
                default: return "";
            }
        }

        //!
        //! \brief returns an appropriate string for prefixing a test result message with the given result
        //!
        static const char* testResultString(TestResult result)
        {
            switch (result)
            {
                case TestResult::kRUNNING: return "RUNNING";
                case TestResult::kPASSED: return "PASSED";
                case TestResult::kFAILED: return "FAILED";
                case TestResult::kWAIVED: return "WAIVED";
                default: return "";
            }
        }

        //!
        //! \brief returns an appropriate output stream (cout or cerr) to use with the given severity
        //!
        static std::ostream& severityOstream(Severity severity)
        {
            //return severity >= Severity::kINFO ? std::cout : std::cerr;
            return std::cout;
        }

        //!
        //! \brief method that implements logging test results
        //!
        static void reportTestResult(const TestAtom& testAtom, TestResult result)
        {
            severityOstream(Severity::kINFO) << "&&&& " << testResultString(result)
                                             << " " << testAtom.mName << " # " << testAtom.mCmdline
                                             << std::endl;
        }

        //!
        //! \brief generate a command line string from the given (argc, argv) values
        //!
        static std::string genCmdlineString(int argc, char const* const* argv)
        {
            std::stringstream ss;
            for (int i = 0; i < argc; i++)
            {
                if (i > 0)
                    ss << " ";
                ss << argv[i];
            }
            return ss.str();
        }

        Severity mReportableSeverity;
    };

    namespace
    {

        //!
        //! \brief produces a LogStreamConsumer object that can be used to log messages of severity kVERBOSE
        //!
        //! Example usage:
        //!
        //!     LOG_VERBOSE(logger) << "hello world" << std::endl;
        //!
        inline LogStreamConsumer LOG_VERBOSE(const Logger& logger)
        {
            return LogStreamConsumer(logger.getReportableSeverity(), Severity::kVERBOSE);
        }

        inline LogStreamConsumer LOG_VERBOSE() {
            return LOG_VERBOSE(gLogger);
        }

        //!
        //! \brief produces a LogStreamConsumer object that can be used to log messages of severity kINFO
        //!
        //! Example usage:
        //!
        //!     LOG_INFO(logger) << "hello world" << std::endl;
        //!
        inline LogStreamConsumer LOG_INFO(const Logger& logger)
        {
            return LogStreamConsumer(logger.getReportableSeverity(), Severity::kINFO);
        }

        inline LogStreamConsumer LOG_INFO() {
            return LOG_INFO(gLogger);
        }

        //!
        //! \brief produces a LogStreamConsumer object that can be used to log messages of severity kWARNING
        //!
        //! Example usage:
        //!
        //!     LOG_WARN(logger) << "hello world" << std::endl;
        //!
        inline LogStreamConsumer LOG_WARN(const Logger& logger)
        {
            return LogStreamConsumer(logger.getReportableSeverity(), Severity::kWARNING);
        }

        inline LogStreamConsumer LOG_WARN() {
            return LOG_WARN(gLogger);
        }

        //!
        //! \brief produces a LogStreamConsumer object that can be used to log messages of severity kERROR
        //!
        //! Example usage:
        //!
        //!     LOG_ERROR(logger) << "hello world" << std::endl;
        //!
        inline LogStreamConsumer LOG_ERROR(const Logger& logger)
        {
            return LogStreamConsumer(logger.getReportableSeverity(), Severity::kERROR);
        }

        inline LogStreamConsumer LOG_ERROR() {
            return LOG_ERROR(gLogger);
        }

        //!
        //! \brief produces a LogStreamConsumer object that can be used to log messages of severity kINTERNAL_ERROR
        //         ("fatal" severity)
        //!
        //! Example usage:
        //!
        //!     LOG_FATAL(logger) << "hello world" << std::endl;
        //!
        inline LogStreamConsumer LOG_FATAL(const Logger& logger)
        {
            return LogStreamConsumer(logger.getReportableSeverity(), Severity::kINTERNAL_ERROR);
        }

        inline LogStreamConsumer LOG_FATAL() {
            return LOG_FATAL(gLogger);
        }


/*        //!
        //! \brief produces a LogStreamConsumer object that can be used to log messages of severity kINTERNAL_ERROR
        //         ("fatal" severity)
        //!
        //! Example usage:
        //!
        //!     LOG_FATAL() << "hello world" << std::endl;
        //!
        inline LogStreamConsumer LOG_CONSOLE()
        {
            return LogStreamConsumer(gLogger.getReportableSeverity(), Severity::kINTERNAL_ERROR);
        }*/


    } // anonymous namespace
}
#endif // RPPRT_LOGGING_H
