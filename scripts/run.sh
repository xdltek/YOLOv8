#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RPP_SERVER_BIN="/usr/local/rpp/bin/rpp_server"

show_help() {
    cat <<'EOF'
Usage: scripts/run.sh [options]

Options:
  -m, --model PATH      ONNX model path. Default: weights/yolov8s.onnx
  -d, --demo NAME       Demo to run: all, rgb, or yuv. Default: all
  -i, --input PATH      RGB image path or I420 YUV path.
  -o, --output PATH     Rendered output path.
  -b, --build-dir DIR   Build directory. Default: build
  -l, --loop COUNT      Measured loop count. Default: 1
      --yuv-width N     I420 width for yuv demo. Default: 1280
      --yuv-height N    I420 height for yuv demo. Default: 720
  -h, --help            Show this help message.
EOF
}

resolve_repo_path() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${ROOT_DIR}/${path}"
    fi
}

is_rpp_server_running() {
    pgrep -f "${RPP_SERVER_BIN}" >/dev/null 2>&1 || pgrep -x "rpp_server" >/dev/null 2>&1
}

ensure_rpp_server() {
    if [[ ! -x "${RPP_SERVER_BIN}" ]]; then
        return 0
    fi

    if is_rpp_server_running; then
        echo "RPP server: already running"
        return 0
    fi

    mkdir -p "${build_dir}/logs"
    local log_path="${build_dir}/logs/rpp_server.log"
    echo "RPP server: starting ${RPP_SERVER_BIN}"
    nohup "${RPP_SERVER_BIN}" > "${log_path}" 2>&1 &
    sleep 1

    if ! is_rpp_server_running; then
        echo "Failed to start rpp_server. See ${log_path}" >&2
        exit 1
    fi
}

run_demo() {
    local selected_demo="$1"
    local selected_input="$2"
    local selected_output="$3"
    local bin
    local args

    case "${selected_demo}" in
        rgb)
            bin="${build_dir}/bin/yolov8_rgb_image_demo"
            [[ -z "${selected_input}" ]] && selected_input="${build_dir}/assets/test.jpg"
            [[ -z "${selected_output}" ]] && selected_output="${build_dir}/output/output_rgb.jpg"
            selected_input="$(resolve_repo_path "${selected_input}")"
            selected_output="$(resolve_repo_path "${selected_output}")"
            args=(-o "${model}" -l "${loop_count}")
            args+=(-i "${selected_input}" --output "${selected_output}")
            ;;
        yuv)
            bin="${build_dir}/bin/yolov8_yuv_frame_demo"
            [[ -z "${selected_input}" ]] && selected_input="${build_dir}/assets/test_0.yuv"
            [[ -z "${selected_output}" ]] && selected_output="${build_dir}/output/output_yuv.jpg"
            selected_input="$(resolve_repo_path "${selected_input}")"
            selected_output="$(resolve_repo_path "${selected_output}")"
            args=(-o "${model}" -l "${loop_count}" --yuv-width "${yuv_width}" --yuv-height "${yuv_height}")
            args+=(--yuv "${selected_input}" --output "${selected_output}")
            ;;
        *)
            echo "--demo must be all, rgb, or yuv." >&2
            exit 1
            ;;
    esac

    echo "Demo: ${selected_demo}"
    echo "Binary: ${bin}"
    echo "Model: ${model}"
    echo "Input: ${selected_input}"
    echo "Output: ${selected_output}"
    "${bin}" "${args[@]}"
}

model="weights/yolov8s.onnx"
demo="all"
input=""
output=""
build_dir="build"
loop_count="1"
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

model="$(resolve_repo_path "${model}")"
build_dir="$(resolve_repo_path "${build_dir}")"

case "${demo}" in
    all)
        if [[ -n "${input}" || -n "${output}" ]]; then
            echo "--input and --output require --demo rgb or --demo yuv when running a single demo." >&2
            exit 1
        fi
        ensure_rpp_server
        run_demo rgb "" ""
        run_demo yuv "" ""
        ;;
    rgb|yuv)
        ensure_rpp_server
        run_demo "${demo}" "${input}" "${output}"
        ;;
    *)
        echo "--demo must be all, rgb, or yuv." >&2
        exit 1
        ;;
esac
