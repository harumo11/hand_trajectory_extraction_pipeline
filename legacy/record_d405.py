#!/usr/bin/env python3
"""
record_d405.py  —  RealSense D405 で RGB + Depth を .bag に無劣化録画する。

設計方針:
  - 生ストリームをそのまま .bag に記録（整列はオフラインで行う）。
  - .bag には intrinsics / depth_scale 等のメタ情報も保存される（再現性）。
  - D405 は torch/CUDA 非依存の pyrealsense2 で動く ＝ Blackwell 問題と無関係。

使い方:
  python legacy/record_d405.py --out ./captures/take01.bag --seconds 20
  （--seconds 省略時は Ctrl+C で停止）

前提:
  pip install pyrealsense2 opencv-python numpy
  ※ Ubuntu24.04 で pip 版が無い場合は librealsense をソース/apt で導入（要確認）。
"""
import argparse
import time
import sys

import numpy as np
import cv2
import pyrealsense2 as rs

# D405 は 1280x720 / 30fps を基本にする（HaWoR の 30fps 系に合わせる）
W, H, FPS = 1280, 720, 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="出力 .bag パス")
    ap.add_argument("--seconds", type=float, default=None, help="録画秒数（省略で Ctrl+C 停止）")
    ap.add_argument("--preview", action="store_true", help="プレビュー表示")
    args = ap.parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    # 生ストリームを有効化（color は OpenCV 互換の bgr8）
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    # .bag へ記録
    config.enable_record_to_file(args.out)

    profile = pipeline.start(config)

    # ---- 重要: depth_scale をデバイスから取得して表示（D405 は D435 と異なる）----
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] depth_scale (m/unit) = {depth_scale}")
    print(f"[INFO] 距離(m) = depth値 * {depth_scale}")
    print(f"[INFO] 録画開始: {args.out}  ({W}x{H}@{FPS})")

    t0 = time.time()
    n = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            n += 1
            if args.preview:
                color = np.asanyarray(frames.get_color_frame().get_data())
                cv2.imshow("D405 color (recording)", color)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.seconds is not None and (time.time() - t0) >= args.seconds:
                break
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C で停止")
    finally:
        pipeline.stop()
        if args.preview:
            cv2.destroyAllWindows()
        dt = time.time() - t0
        print(f"[INFO] 録画終了: {n} frames / {dt:.1f}s (実効 {n/max(dt,1e-6):.1f} fps)")
        print(f"[INFO] 次は legacy/split_bag.py で RGB(mp4) と Depth(PNG16)+intrinsics に分離してください。")


if __name__ == "__main__":
    sys.exit(main())
