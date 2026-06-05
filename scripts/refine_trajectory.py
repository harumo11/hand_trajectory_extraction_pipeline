#!/usr/bin/env python3
"""
refine_trajectory.py — 2D キーポイントによる手並進リファインの CLI（CLAUDE.md §7 の精度改善）。

前提:
  1. パイプライン実行済み（<take>/rgb/ と <take>/world_trajectory.npz が存在）
  2. 2D キーポイント検出済み: <take>/keypoints_2d.npz
     （無ければ: .venv_mp/bin/python tools/detect_keypoints_2d.py --video <take>/rgb.mp4
                  --out <take>/keypoints_2d.npz   ※隔離 venv は tools/setup_keypoint_env.sh で構築）

処理:
  フレーム×手ごとの並進補正 Δt を最適化（実体は handtraj.refine.refine_take）し、
  world_trajectory.npz をリファイン済みに更新（追加キー: refined / delta_t_world）。
  併せて refine_report.json と比較用オーバーレイ（overlay_refined/）を出力する。

使い方:
  python scripts/refine_trajectory.py --take captures/take03
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))  # プロジェクトルート
from handtraj.refine import refine_take  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", required=True, help="テイクディレクトリ")
    ap.add_argument("--keypoints", default=None, help="keypoints_2d.npz（既定 <take>/keypoints_2d.npz）")
    ap.add_argument("--sigma_vel", type=float, default=0.003,
                    help="時間平滑化の強さ [m/frame]（小さいほど滑らか）")
    ap.add_argument("--gate_px", type=float, default=80.0, help="誤検出ゲートの残差閾値")
    ap.add_argument("--skip_overlay", action="store_true", help="比較オーバーレイ動画を生成しない")
    args = ap.parse_args()

    try:
        refine_take(args.take, keypoints=args.keypoints, sigma_vel_m=args.sigma_vel,
                    gate_px=args.gate_px, skip_overlay=args.skip_overlay)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
