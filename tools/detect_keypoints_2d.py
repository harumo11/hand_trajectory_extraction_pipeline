#!/usr/bin/env python3
"""
detect_keypoints_2d.py — MediaPipe HandLandmarker で動画全フレームの手 2D キーポイントを検出する。

★必ず隔離 venv（.venv_mp）の python で実行すること★
  mediapipe は numpy>=2 を要求し本体 .venv と非互換（tools/setup_keypoint_env.sh で構築）。
  本スクリプトは handtraj に依存しない（cv2 / mediapipe / numpy のみ）。

出力 npz:
  k2d      [T, 2, 21, 2] float32 … 画素座標。軸1は [left, right]（MediaPipe handedness ラベル準拠）。
                                    未検出は NaN
  detected [T, 2] bool            … 検出フラグ
  score    [T, 2] float32         … handedness スコア（未検出は 0）
  width, height, n_frames        … メタ情報

関節順序は OpenPose/MANO 互換（0=手首, 1-4=親指, …, 17-20=小指）= 本体側と同一。
左右ラベルの取り違えは本体側（handtraj.refine）が初期推定との距離で補正するため、
ここでは MediaPipe のラベルをそのまま記録する。

使い方:
  .venv_mp/bin/python tools/detect_keypoints_2d.py --video <take>/rgb.mp4 --out <take>/keypoints_2d.npz
"""
import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
HAND_IDX = {"left": 0, "right": 1}


def model_path():
    cache = os.path.expanduser("~/.cache/handtraj")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "hand_landmarker.task")
    if not os.path.exists(path):
        print(f"[INFO] モデルをダウンロード: {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_conf", type=float, default=0.3, help="検出信頼度の下限")
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path()),
        running_mode=vision.RunningMode.VIDEO,    # 時間的に安定なトラッキングモード
        num_hands=2,
        min_hand_detection_confidence=args.min_conf,
        min_tracking_confidence=args.min_conf,
    ))

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    k2d_all, det_all, score_all = [], [], []
    t = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        ts_ms = int(round(t * 1000.0 / fps))
        res = det.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), ts_ms)
        k2d = np.full((2, 21, 2), np.nan, np.float32)
        dt = np.zeros(2, bool)
        sc = np.zeros(2, np.float32)
        for lmks, handed in zip(res.hand_landmarks, res.handedness):
            h = HAND_IDX.get(handed[0].category_name.lower())
            if h is None:
                continue
            s = float(handed[0].score)
            if dt[h] and s <= sc[h]:
                continue        # 同ラベル重複はスコアが高い方を採用
            k2d[h] = [[l.x * W, l.y * H] for l in lmks]
            dt[h] = True
            sc[h] = s
        k2d_all.append(k2d)
        det_all.append(dt)
        score_all.append(sc)
        t += 1
        if t % 200 == 0:
            print(f"[INFO] {t} frames...")
    cap.release()

    k2d_arr = np.stack(k2d_all)
    det_arr = np.stack(det_all)
    np.savez(args.out,
             k2d=k2d_arr, detected=det_arr, score=np.stack(score_all),
             width=W, height=H, n_frames=t)
    print(f"[INFO] 検出完了: {t} frames -> {args.out}")
    print(f"[INFO] 検出率: left {det_arr[:,0].mean()*100:.0f}%  right {det_arr[:,1].mean()*100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
