#!/usr/bin/env python3
"""
record_rgb.py — UVC/Web カメラから RGB を mp4 に録画する（CLAUDE.md §6-2）。

HaWoR は 30fps 前提（フレーム抽出時に fps=30 で再サンプル）のため、30fps で録画し、
実効 fps が大きくずれた場合は警告する。

使い方:
  python record_rgb.py --out captures/take01/rgb.mp4 --seconds 20 --preview
  python record_rgb.py --device 2 --size 1280x720 --fps 30 --out take.mp4
  （--seconds 省略時は Ctrl+C か preview ウィンドウで 'q'）

備考:
  ライブ録画は遅延・取りこぼしを避けるため cv2.VideoWriter を直接使う
  （handtraj.video_io.Mp4Writer の close 時 ffmpeg 再エンコードは収録後の整形向き）。
"""
import argparse
import os
import sys
import time

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="出力 mp4 パス")
    ap.add_argument("--device", type=int, default=0, help="VideoCapture デバイス番号")
    ap.add_argument("--size", default="1280x720", help="解像度 WxH")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    W, H = map(int, args.size.lower().split("x"))
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"[ERROR] カメラ {args.device} を開けません（/dev/video* と権限を確認）")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    W_act = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_act = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (W_act, H_act) != (W, H):
        print(f"[WARN] 要求解像度 {W}x{H} は非対応。{W_act}x{H_act} で録画します。")
        W, H = W_act, H_act

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    print(f"[INFO] 録画開始: {args.out} ({W}x{H} 目標{args.fps}fps)  "
          f"{'preview q で停止' if args.preview else 'Ctrl+C で停止'}")
    t0 = time.time()
    n = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] フレーム取得失敗")
                break
            writer.write(frame)
            n += 1
            if args.preview:
                cv2.imshow("recording (q to stop)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.seconds is not None and (time.time() - t0) >= args.seconds:
                break
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C で停止")
    finally:
        dt = time.time() - t0
        cap.release()
        writer.release()
        if args.preview:
            cv2.destroyAllWindows()
        eff = n / max(dt, 1e-6)
        print(f"[INFO] 録画終了: {n} frames / {dt:.1f}s (実効 {eff:.1f} fps)")
        if abs(eff - args.fps) > args.fps * 0.1:
            print(f"[WARN] 実効 fps が目標 {args.fps} から 10% 以上ずれています。"
                  f"mp4 のタイムスタンプと HaWoR の 30fps 再サンプルで時間軸が歪む可能性。"
                  f"露出設定・USB帯域・解像度を見直してください。")
        print(f"[INFO] 次: python run_pipeline.py --video {args.out} --take <出力dir> "
              f"[--intrinsics intrinsics.json]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
