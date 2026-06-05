#!/usr/bin/env python3
"""video_io.py — 動画入出力の共通ヘルパ（mp4 書き出し / fps・サイズ取得 / フレーム読み出し）。

Mp4Writer は旧 tools/video_util.py の実装をそのまま移設したもの（挙動変更禁止）。
video_fps / video_size は旧 run_pipeline.py のヘルパを移設。
"""
import os
import shutil
import subprocess

import cv2


class Mp4Writer:
    """cv2.VideoWriter の薄いラッパ。close() 時に ffmpeg があれば H.264 (crf23) に再エンコードする。"""

    def __init__(self, out_path, fps, size_wh):
        self.out_path = out_path
        self.raw_path = out_path + ".raw.mp4"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        self.writer = cv2.VideoWriter(self.raw_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, tuple(size_wh))
        self.n = 0

    def write(self, bgr_frame):
        self.writer.write(bgr_frame)
        self.n += 1

    def close(self):
        self.writer.release()
        if shutil.which("ffmpeg"):
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", self.raw_path,
                                "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                                self.out_path])
            if r.returncode == 0:
                os.remove(self.raw_path)
                return
            print("[WARN] H.264 再エンコード失敗。mp4v のまま出力します。")
        os.replace(self.raw_path, self.out_path)


def video_fps(path):
    """動画の fps を返す（取得失敗時は None）。"""
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps and fps > 0 else None
    except Exception:
        return None


def video_size(path):
    """動画の解像度 (W, H) を返す。"""
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def read_frame(path, idx):
    """指定インデックスのフレーム（BGR ndarray）を返す。取得できなければ None。"""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def iter_frames(path):
    """動画を先頭から順次走査し (idx, frame) を yield する。"""
    cap = cv2.VideoCapture(path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield idx, frame
        idx += 1
    cap.release()
