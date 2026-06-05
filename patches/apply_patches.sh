#!/usr/bin/env bash
# apply_patches.sh — HaWoR クローン後に vendored コードへ必須パッチを適用する。
#
# 内容（README「vendored コードへの最小パッチ」表に対応）:
#   01: DROID-SLAM setup.py の -gencode ハードコード削除（TORCH_CUDA_ARCH_LIST 有効化, RTX 5080/sm_120 必須）
#       + correlation_kernels.cu / altcorr_kernel.cu の .type() → .scalar_type()（torch 2.x 対応）
#   02: lietorch の DISPATCH マクロ .type() → .scalar_type()（torch 2.x 対応）
#
# 使い方（third_party/HaWoR を --recursive クローンした後に一度だけ）:
#   bash patches/apply_patches.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

apply() {
  local repo="$1" patch="$2"
  if git -C "$repo" apply --reverse --check "$patch" 2>/dev/null; then
    echo "[SKIP] $(basename "$patch") は適用済み"
  elif git -C "$repo" apply --check "$patch" 2>/dev/null; then
    git -C "$repo" apply "$patch"
    echo "[OK] $(basename "$patch")"
  else
    echo "[ERROR] $(basename "$patch") を適用できません（git -C $repo status で確認）" >&2
    exit 1
  fi
}

apply "$ROOT/third_party/HaWoR" "$ROOT/patches/01_droid_slam_sm120_torch2x.patch"
apply "$ROOT/third_party/HaWoR/thirdparty/DROID-SLAM/thirdparty/lietorch" "$ROOT/patches/02_lietorch_torch2x.patch"
echo "[DONE] パッチ適用完了。次: DROID-SLAM 拡張のビルド（README セットアップ手順 6）"
