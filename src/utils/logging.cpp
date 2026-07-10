/** @file logging.cpp
 *
 * @brief Process-wide log file path helpers for stream-style logging.
 */
#include <string>
#include "logging.h"

namespace sample {

    //! Optional file path used by the common stream logger.
    std::string log_path("");


    //! Return the current optional log file path.
    const std::string &get_log_file_path() {
        return log_path;
    }

    //! Set the optional file path used by stream logger output.
    void set_log_file_path_111(const std::string &path) {
        log_path = path;
    }
}
