#!/usr/bin/env python3
"""
visualize_3d.py — world 座標系の 3D 可視化動画を生成する（matplotlib・ヘッドレス対応）。

描画内容（毎フレーム）:
  - カメラ軌跡（全体: 薄灰 / 現在まで: 青）と現在のカメラ姿勢（フラスタム＋光軸）
  - 両手の 21 関節スケルトン（左=赤, 右=緑。指ごとのボーン描画で関節の屈曲が見える）
  - 手首の軌跡（薄い色、現在まで）

座標は world_trajectory.npz の joints（= α 適用済み・メートル）と、HaWoR の SLAM カメラ軌道
（同じ α でスケール）。表示は HaWoR demo と同じ R_x=diag(1,-1,-1) を掛けた y-up 系
（--no_flip で生データ軸のまま）。

使い方:
  python tools/visualize_3d.py --hawor <take>/rgb --trajectory <take>/world_trajectory.npz \
                               --out <take>/vis3d/world_3d.mp4 [--fps 30] [--stride 1] \
                               [--size 1280x720] [--elev 20] [--azim -60]

実体は handtraj.visualization.Skeleton3DRenderer に移設済み。本スクリプトはその薄いラッパ。
"""
import argparse
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_THIS_DIR, "..")))
from handtraj.hawor_adapter import HaworSequence  # noqa: E402
from handtraj.visualization import Skeleton3DRenderer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hawor", required=True, help="HaWoR seq_folder（カメラ軌道の取得元）")
    ap.add_argument("--trajectory", required=True, help="world_trajectory.npz（α適用済み関節）")
    ap.add_argument("--out", required=True, help="出力 mp4 パス")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--stride", type=int, default=1, help="描画フレームの間引き（動画 fps は fps/stride）")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--no_flip", action="store_true", help="表示用 R_x 変換を行わない（生 world 軸）")
    args = ap.parse_args()

    d = np.load(args.trajectory)
    joints, valid = d["joints"].astype(np.float64), d["valid_per_hand"]
    alpha = float(d["alpha"])

    seq = HaworSequence(args.hawor)
    R_c2w = seq.R_c2w
    cam_pos = seq.cam_centers * alpha   # 関節と同じ α でスケール

    W, H = map(int, args.size.lower().split("x"))

    renderer = Skeleton3DRenderer(joints, valid, cam_pos, R_c2w, flip=not args.no_flip)
    n = renderer.render_video(args.out, fps=args.fps, stride=args.stride,
                              size=(W, H), elev=args.elev, azim=args.azim)
    out_fps = max(args.fps / args.stride, 1.0)
    print(f"[INFO] 3D 可視化動画: {args.out} ({n} frames, {out_fps:.2f} fps, alpha={alpha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
