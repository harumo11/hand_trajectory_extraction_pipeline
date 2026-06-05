#!/usr/bin/env python3
"""
split_bag.py  —  録画した .bag を「オフラインで」整列・分離する。

出力:
  - RGB:        <out>/rgb.mp4              … HaWoR demo.py の --video_path に渡す
  - Depth:      <out>/depth/depth_000000.png (16bit, 整列済み) … M3 の実寸較正用
  - メタ情報:    <out>/intrinsics.json      … fx,fy,cx,cy,fps,depth_scale 等

設計上の要点:
  - playback.set_real_time(False) で「フレームを落とさず」全フレーム処理。
  - rs.align(color) で Depth を RGB 画素に整列（後段の逆投影/較正のため）。
  - depth_scale は .bag のデバイスから取得（D405 は D435 と異なるため必須）。
  - mp4 は Depth を保持できないので Depth は 16bit PNG で別保存（無効画素は 0 のまま）。

使い方:
  python legacy/split_bag.py --bag ./captures/take01.bag --out ./captures/take01
"""
import argparse
import json
import os
import sys

import numpy as np
import cv2
import pyrealsense2 as rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="入力 .bag")
    ap.add_argument("--out", required=True, help="出力ディレクトリ")
    args = ap.parse_args()

    depth_dir = os.path.join(args.out, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(args.bag, repeat_playback=False)
    profile = pipe.start(cfg)

    # 全フレームを確実に処理（リアルタイム再生だと間引かれる）
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    writer = None
    intr_saved = False
    fps = 30  # 既定。下で実ストリームの fps に上書きを試みる
    n = 0
    last_idx = -1

    try:
        while True:
            try:
                frames = pipe.wait_for_frames()
            except RuntimeError:
                break  # 再生終了

            # 再生のループ/巻き戻り検知（保険）
            idx = frames.get_frame_number()
            if idx < last_idx:
                break
            last_idx = idx

            aligned = align.process(frames)
            cframe = aligned.get_color_frame()
            dframe = aligned.get_depth_frame()
            if not cframe or not dframe:
                continue

            color = np.asanyarray(cframe.get_data())           # HxWx3 (bgr8)
            depth = np.asanyarray(dframe.get_data())            # HxW (uint16)

            if not intr_saved:
                vsp = cframe.profile.as_video_stream_profile()
                intr = vsp.get_intrinsics()
                try:
                    fps = int(round(vsp.fps()))
                except Exception:
                    pass
                meta = {
                    "width": intr.width,
                    "height": intr.height,
                    "fx": intr.fx, "fy": intr.fy,
                    "cx": intr.ppx, "cy": intr.ppy,
                    "model": str(intr.model),
                    "coeffs": list(intr.coeffs),
                    "fps": fps,
                    "depth_scale_m_per_unit": depth_scale,   # ← M3 で 距離(m)=PNG値*これ
                    "note": "depth PNG は整列済み・16bit。0 は無効画素(穴)。"
                }
                with open(os.path.join(args.out, "intrinsics.json"), "w") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
                intr_saved = True

                h, w = color.shape[:2]
                writer = cv2.VideoWriter(
                    os.path.join(args.out, "rgb.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )

            writer.write(color)
            cv2.imwrite(os.path.join(depth_dir, f"depth_{n:06d}.png"), depth)  # 16bit PNG
            n += 1
    finally:
        pipe.stop()
        if writer is not None:
            writer.release()
        print(f"[INFO] 分離完了: {n} frames")
        print(f"[INFO] RGB  : {os.path.join(args.out, 'rgb.mp4')}")
        print(f"[INFO] Depth: {depth_dir}/depth_*.png (16bit, 整列済み)")
        print(f"[INFO] Meta : {os.path.join(args.out, 'intrinsics.json')}")
        print(f"[INFO] HaWoR: python demo.py --video_path {os.path.join(args.out,'rgb.mp4')} --vis_mode world")


if __name__ == "__main__":
    sys.exit(main())
