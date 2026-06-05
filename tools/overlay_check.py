#!/usr/bin/env python3
"""
overlay_check.py — 関門③: MANO メッシュ投影を RGB に重ねてカメラ規約を確認する。

アダプタの verts_cam_m（カメラ座標・OpenCV系仮定）を K で投影し、RGB へ重畳描画する。
手の輪郭と投影点群が一致すれば、カメラ規約（+Z 前方・y 下・u=fx·X/Z+cx）の仮定は正しい。
ズレ方（上下反転・左右反転）から座標系の誤りを診断できる。

出力:
  - <out>/overlay_NNNNNN.png … 等間隔サンプルの静止画（--num 枚、目視確認用）
  - <out>/overlay.mp4        … 全フレームの重畳動画（--video_out で変更、--no_video で省略）

使い方:
  python tools/overlay_check.py --hawor <take>/rgb --video <take>/rgb.mp4 \
                                --intrinsics <take>/intrinsics.json --out <take>/overlay \
                                [--num 6] [--video_out <path>.mp4 | --no_video]

実体は handtraj.visualization.OverlayRenderer に移設済み。本スクリプトはその薄いラッパ。
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_THIS_DIR, "..")))
from handtraj.camera import Intrinsics  # noqa: E402
from handtraj.hawor_adapter import HaworSequence  # noqa: E402
from handtraj.visualization import OverlayRenderer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hawor", required=True, help="HaWoR seq_folder")
    ap.add_argument("--video", required=True, help="rgb.mp4")
    ap.add_argument("--intrinsics", required=True, help="intrinsics.json")
    ap.add_argument("--out", required=True, help="出力ディレクトリ")
    ap.add_argument("--num", type=int, default=6, help="静止画のフレーム数")
    ap.add_argument("--video_out", default=None,
                    help="重畳動画の出力パス（既定 <out>/overlay.mp4）")
    ap.add_argument("--no_video", action="store_true", help="動画出力を省略（静止画のみ）")
    args = ap.parse_args()

    intr = Intrinsics.load(args.intrinsics)
    seq = HaworSequence(args.hawor)
    if not seq.valid.any(axis=1).any():
        print("[ERROR] valid フレームがありません")
        return 1

    renderer = OverlayRenderer(seq, intr)
    n_drawn = renderer.snapshots(args.video, args.out, args.num)

    if not args.no_video:
        video_out = args.video_out or os.path.join(args.out, "overlay.mp4")
        renderer.render_video(args.video, video_out)

    print(f"[INFO] 静止画 {n_drawn} 枚出力。手の輪郭と点群の一致を目視確認してください（関門③）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
