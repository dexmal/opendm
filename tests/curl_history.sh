#!/usr/bin/env bash
# Example /v1/infer request with observation.history_images (oldest first).
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:7891/v1/infer}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG_DIR="${SCRIPT_DIR}/../assets/demo/images/episode0"

# Current frame
CUR_FRAME="${CUR_FRAME:-14}"
# History frames (oldest first). Typical: head camera / cam_high only.
HISTORY_FRAMES=(9 10 11 12 13)

encode_image() {
  base64 <"$1" | tr -d '\n'
}

HEAD_IMAGE="$(encode_image "${IMG_DIR}/cam_high/${CUR_FRAME}.jpg")"
LEFT_IMAGE="$(encode_image "${IMG_DIR}/cam_left_wrist/${CUR_FRAME}.jpg")"
RIGHT_IMAGE="$(encode_image "${IMG_DIR}/cam_right_wrist/${CUR_FRAME}.jpg")"

HISTORY_JSON="["
for i in "${!HISTORY_FRAMES[@]}"; do
  frame="${HISTORY_FRAMES[$i]}"
  b64="$(encode_image "${IMG_DIR}/cam_high/${frame}.jpg")"
  if [[ "${i}" -gt 0 ]]; then
    HISTORY_JSON+=","
  fi
  HISTORY_JSON+="\"${b64}\""
done
HISTORY_JSON+="]"

curl -sS -X POST "${BASE_URL}" \
  -H 'Content-Type: application/json' \
  --data @- <<EOF
{
  "observation": {
    "prompt": "Hold the roller for smoothing materials with both arms to pick it up",
    "robot_type": "DOS W1",
    "state": [-0.2, -0.16923, -0.13846, -0.10769, -0.07692, -0.04615, -0.01538, 0.01538, 0.04615, 0.07692, 0.10769, 0.13846, 0.16923, 0.2],
    "images": {
      "1": "${HEAD_IMAGE}",
      "2": "${LEFT_IMAGE}",
      "3": "${RIGHT_IMAGE}"
    },
    "history_images": ${HISTORY_JSON}
  }
}
EOF
echo
