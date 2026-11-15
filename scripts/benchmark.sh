#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${YOLO_COLOR:-always}" != "never" ]]; then
    BLUE=$'\033[1;34m'; CYAN=$'\033[1;36m'; GREEN=$'\033[1;32m'; RED=$'\033[1;31m'; RESET=$'\033[0m'
else
    BLUE=""; CYAN=""; GREEN=""; RED=""; RESET=""
fi

show_help() {
    cat <<'EOF'
Usage: scripts/benchmark.sh [options]

Options:
  -m, --model PATH      ONNX model path. Required.
  -d, --demo NAME       Workload selector: all, rgb, or yuv. Default: all
  -i, --input PATH      RGB image path or I420 YUV path.
  -o, --output PATH     Rendered output path.
  -b, --build-dir DIR   Build directory. Default: build
  -l, --loop COUNT      Measured loop count. Default: 20
      --yuv-width N     I420 width for yuv demo. Default: 1280
      --yuv-height N    I420 height for yuv demo. Default: 720
  -h, --help            Show this help message.

Logs are saved to <build-dir>/logs/benchmark_<workload>_<timestamp>.log.
EOF
}

model=""
demo="all"
input=""
output=""
build_dir="build"
loop_count="20"
yuv_width="1280"
yuv_height="720"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            model="$2"
            shift 2
            ;;
        -d|--demo|--demo-type)
            demo="$2"
            shift 2
            ;;
        -i|--input)
            input="$2"
            shift 2
            ;;
        -o|--output)
            output="$2"
            shift 2
            ;;
        -b|--build-dir)
            build_dir="$2"
            shift 2
            ;;
        -l|--loop)
            loop_count="$2"
            shift 2
            ;;
        --yuv-width)
            yuv_width="$2"
            shift 2
            ;;
        --yuv-height)
            yuv_height="$2"
            shift 2
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

if [[ ! "${loop_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${RED}[ERROR   ]${RESET} --loop must be a positive integer." >&2
    exit 1
fi

if [[ "${demo}" != "all" && "${demo}" != "rgb" && "${demo}" != "yuv" ]]; then
    echo "--demo must be all, rgb, or yuv." >&2
    exit 1
fi

if [[ "${demo}" == "all" && ( -n "${input}" || -n "${output}" ) ]]; then
    echo "--input and --output require --demo rgb or --demo yuv when benchmarking a single workload." >&2
    exit 1
fi

if [[ -z "${model}" ]]; then
    echo "${RED}[ERROR   ]${RESET} No ONNX model was specified." >&2
    echo "${CYAN}[DETAIL  ]${RESET} This repository does not include a model." >&2
    echo "${CYAN}[ACTION  ]${RESET} Provide one with: scripts/benchmark.sh --model /path/to/model.onnx" >&2
    exit 1
fi

if [[ "${build_dir}" == /* ]]; then
    resolved_build_dir="${build_dir}"
else
    resolved_build_dir="${ROOT_DIR}/${build_dir}"
fi
log_dir="${resolved_build_dir}/logs"
mkdir -p "${log_dir}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="${log_dir}/benchmark_${demo}_${timestamp}.log"

cmd=("${SCRIPT_DIR}/run.sh" --demo "${demo}" --loop "${loop_count}" --build-dir "${build_dir}")
[[ -n "${model}" ]] && cmd+=(--model "${model}")
[[ -n "${input}" ]] && cmd+=(--input "${input}")
[[ -n "${output}" ]] && cmd+=(--output "${output}")
if [[ "${demo}" == "yuv" || "${demo}" == "all" ]]; then
    cmd+=(--yuv-width "${yuv_width}" --yuv-height "${yuv_height}")
fi

{
    printf '%s' "${BLUE}"
    cat <<EOF
+------------------------------------------------------------------+
|             XDL YOLOv8 PERFORMANCE BENCHMARK                     |
+------------------------------------------------------------------+
  Test scope               : ${demo^^}
  Measured frames          : ${loop_count} per workload (+ 1 warmup)
  Performance log          : ${log_path}
+------------------------------------------------------------------+
EOF
    printf '%s' "${RESET}"
    echo "${CYAN}[BENCH   ]${RESET} Benchmark started; live progress follows below."
    "${cmd[@]}"
    echo "${GREEN}[ARCHIVE ]${RESET} Benchmark log saved: ${log_path}"
} 2>&1 | tee "${log_path}"
