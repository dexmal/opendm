#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:7891/v1/infer}"
ROBOT_TYPE="${2:-DOS W1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG_DIR="${SCRIPT_DIR}/../assets/demo/images/episode0"

encode_image() {
  base64 <"$1" | tr -d '\n'
}

HEAD_IMAGE="$(encode_image "${IMG_DIR}/cam_high/0.jpg")"
LEFT_IMAGE="$(encode_image "${IMG_DIR}/cam_left_wrist/0.jpg")"
RIGHT_IMAGE="$(encode_image "${IMG_DIR}/cam_right_wrist/0.jpg")"

curl -sS -X POST "${BASE_URL}" \
  -H 'Content-Type: application/json' \
  --data @- <<EOF
{
  "observation": {
    "prompt": "Hold the roller for smoothing materials with both arms to pick it up",
    "robot_type": "${ROBOT_TYPE}",
    "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "images": {
      "1": "${HEAD_IMAGE}",
      "2": "${LEFT_IMAGE}",
      "3": "${RIGHT_IMAGE}"
    }
  }
}
EOF
echo
