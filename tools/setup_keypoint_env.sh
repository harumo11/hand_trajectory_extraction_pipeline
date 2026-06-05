#!/usr/bin/env bash
# setup_keypoint_env.sh — 2D キーポイント検出用の隔離 venv (.venv_mp) を作成する。
#
# mediapipe は numpy>=2 を要求し、本体 .venv（numpy<2 必須・HaWoR 互換）と同居できないため、
# 検出だけを別 venv + サブプロセスで実行する（CLAUDE.md §4 の注意参照）。
#
# 使い方:
#   bash tools/setup_keypoint_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv_mp"

if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import mediapipe" 2>/dev/null; then
    echo "[INFO] $VENV は構築済み"
    exit 0
fi
python3.12 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet mediapipe opencv-python
"$VENV/bin/python" -c "import mediapipe, cv2, numpy; print('[INFO] .venv_mp OK: mediapipe', mediapipe.__version__, '/ numpy', numpy.__version__)"
echo "[INFO] 次: .venv_mp/bin/python tools/detect_keypoints_2d.py --video <rgb.mp4> --out <keypoints_2d.npz>"
