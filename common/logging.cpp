#include <string>
#include "logging.h"

namespace sample {

    std::string log_path("");


    const std::string &get_log_file_path() {
        return log_path;
    }

    void set_log_file_path_111(const std::string &path) {
        log_path = path;
    }
}