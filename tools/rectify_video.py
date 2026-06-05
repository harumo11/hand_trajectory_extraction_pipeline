#!/usr/bin/env python3
"""
rectify_video.py — 動画を理想ピンホール（主点=画像中央・fx=fy・歪みゼロ）へレクティフィケーションする。

背景（CLAUDE.md §2/§7）:
  HaWoR は内部で「焦点距離 fx=fy・主点=画像中央・歪みなし」を仮定して手の 3D 位置を推定する。
  実カメラの主点が中央からずれている／歪みがあると、その分が world 出力の系統誤差になる
  （例: 主点 18px ずれ × 手の距離 0.5m ≈ 0.9cm）。
  本ツールでキャリブレーション値に基づき動画を変換しておけば、HaWoR の仮定が現実と一致する。

使い方:
  python tools/rectify_video.py --video rgb_raw.mp4 --intrinsics intrinsics.json \
                                --out_video rgb.mp4 --out_intrinsics intrinsics_rect.json

実装は handtraj.camera.VideoRectifier / Intrinsics に移設済み。本ファイルは薄いラッパ。
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))
from handtraj.camera import Intrinsics, VideoRectifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--out_intrinsics", required=True)
    args = ap.parse_args()

    intr = Intrinsics.load(args.intrinsics)
    if not intr.needs_rectify():
        print("[INFO] 補正不要（既に理想ピンホール相当）。")
        return 0
    _, intr_rect = VideoRectifier(intr).rectify_video(args.video, args.out_video)
    intr_rect.save(args.out_intrinsics)
    print(f"[INFO] 新 intrinsics: {args.out_intrinsics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
