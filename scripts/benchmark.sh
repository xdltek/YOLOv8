#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

show_help() {
    cat <<'EOF'
Usage: scripts/benchmark.sh [options]

Options:
  -m, --model PATH      ONNX model path. Default: weights/yolov8s.onnx
  -d, --demo NAME       Demo to run: all, rgb, or yuv. Default: all
  -i, --input PATH      RGB image path or I420 YUV path.
  -l, --loop COUNT      Measured loop count. Default: 20
      --yuv-width N     I420 width for yuv demo. Default: 1280
      --yuv-height N    I420 height for yuv demo. Default: 720
  -h, --help            Show this help message.

Logs are saved to build/logs/benchmark_<demo>_<timestamp>.log.
EOF
}

model="weights/yolov8s.onnx"
demo="all"
input=""
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

if [[ "${demo}" != "all" && "${demo}" != "rgb" && "${demo}" != "yuv" ]]; then
    echo "--demo must be all, rgb, or yuv." >&2
    exit 1
fi

if [[ "${demo}" == "all" && -n "${input}" ]]; then
    echo "--input requires --demo rgb or --demo yuv when benchmarking a single demo." >&2
    exit 1
fi

log_dir="${ROOT_DIR}/build/logs"
mkdir -p "${log_dir}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="${log_dir}/benchmark_${demo}_${timestamp}.log"

cmd=("${SCRIPT_DIR}/run.sh" --model "${model}" --demo "${demo}" --loop "${loop_count}")
[[ -n "${input}" ]] && cmd+=(--input "${input}")
if [[ "${demo}" == "yuv" || "${demo}" == "all" ]]; then
    cmd+=(--yuv-width "${yuv_width}" --yuv-height "${yuv_height}")
fi

{
    echo "Benchmark demo: ${demo}"
    echo "Measured loop count: ${loop_count}"
    echo "Log: ${log_path}"
    "${cmd[@]}"
} 2>&1 | tee "${log_path}"
