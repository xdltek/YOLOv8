#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CMAKE_GENERATOR_NAME="Unix Makefiles"

show_help() {
    cat <<'EOF'
Usage: scripts/build.sh [options]

Options:
  -b, --build-dir DIR   Build directory. Default: build
      --perf            Enable YOLO_ENABLE_RPP_PERF.
  -h, --help            Show this help message.
EOF
}

resolve_repo_path() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        readlink -m "${path}"
    else
        readlink -m "${ROOT_DIR}/${path}"
    fi
}

clean_build_outputs() {
    if [[ "${build_dir}" == "/" || "${build_dir}" == "${ROOT_DIR}" ]]; then
        echo "Refusing to clean unsafe build directory: ${build_dir}" >&2
        exit 1
    fi

    if [[ ! -d "${build_dir}" ]]; then
        return 0
    fi

    echo "Cleaning previous build outputs in ${build_dir}"
    rm -rf "${build_dir}/CMakeCache.txt" \
           "${build_dir}/CMakeFiles" \
           "${build_dir}/CTestTestfile.cmake" \
           "${build_dir}/Testing" \
           "${build_dir}/Makefile" \
           "${build_dir}/build.ninja" \
           "${build_dir}/rules.ninja" \
           "${build_dir}/.ninja_deps" \
           "${build_dir}/.ninja_log" \
           "${build_dir}/cmake_install.cmake" \
           "${build_dir}/compile_commands.json" \
           "${build_dir}/install_manifest.txt" \
           "${build_dir}/bin" \
           "${build_dir}/lib" \
           "${build_dir}/assets" \
           "${build_dir}/output" \
           "${build_dir}/trace" \
           "${build_dir}/logs"
}

ensure_compatible_build_dir() {
    local cache_file="${build_dir}/CMakeCache.txt"
    if [[ ! -f "${cache_file}" ]]; then
        return 0
    fi

    local cached_generator
    cached_generator="$(awk -F= '/^CMAKE_GENERATOR:INTERNAL=/{print $2}' "${cache_file}")"
    if [[ -z "${cached_generator}" || "${cached_generator}" == "${CMAKE_GENERATOR_NAME}" ]]; then
        return 0
    fi

    echo "Existing build directory uses CMake generator '${cached_generator}', recreating CMake cache for '${CMAKE_GENERATOR_NAME}'."
    rm -rf "${build_dir}/CMakeCache.txt" \
           "${build_dir}/CMakeFiles" \
           "${build_dir}/build.ninja" \
           "${build_dir}/rules.ninja" \
           "${build_dir}/.ninja_deps" \
           "${build_dir}/.ninja_log" \
           "${build_dir}/cmake_install.cmake" \
           "${build_dir}/Makefile"
}

build_dir="build"
enable_perf="OFF"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build-dir)
            build_dir="$2"
            shift 2
            ;;
        --perf)
            enable_perf="ON"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

build_dir="$(resolve_repo_path "${build_dir}")"
clean_build_outputs
ensure_compatible_build_dir
cmake -S "${ROOT_DIR}" -B "${build_dir}" -G "${CMAKE_GENERATOR_NAME}" -DYOLO_ENABLE_RPP_PERF="${enable_perf}"
cmake --build "${build_dir}" -j"$(nproc)"
