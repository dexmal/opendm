#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:7891/process_frame}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG_DIR="${SCRIPT_DIR}/../assets/demo/images/episode0"

curl -sS -X POST "${BASE_URL}" \
  -F 'text=Hold the roller for smoothing materials with both arms to pick it up' \
  -F 'robot_type=DOS W1' \
  -F 'states=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]' \
  -F "image=@${IMG_DIR}/cam_high/0.jpg" \
  -F "image=@${IMG_DIR}/cam_left_wrist/0.jpg" \
  -F "image=@${IMG_DIR}/cam_right_wrist/0.jpg"
echo
