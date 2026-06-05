#!/usr/bin/env python3
"""camera.py — カメラ内部パラメータと、それに紐づく処理（投影・レクティファイ・キャリブレーション）。

- Intrinsics      : intrinsics.json と往復できる内部パラメータの dataclass。
- project         : OpenCV系カメラ座標 (m) を K で画素へ投影。
- VideoRectifier  : 動画を理想ピンホール化（旧 tools/rectify_video.py のロジックを移設）。
- CameraCalibrator: チェスボードキャリブレーション（旧 calibrate_camera.py のロジックを移設）。

カメラ規約は OpenCV系（+Z 前方・y 下向き・u=fx*X/Z+cx）を仮定する（CLAUDE.md §5）。
"""
import glob
import json
from dataclasses import dataclass, field

import cv2
import numpy as np

# Intrinsics の「既知キー」。JSON から読むとき、これ以外のキーは extra へ退避する。
_KNOWN_KEYS = ("width", "height", "fx", "fy", "cx", "cy", "model", "coeffs", "fps")


@dataclass
class Intrinsics:
    """カメラ内部パラメータ。intrinsics.json と往復する。

    extra は calib_rms_px / calib_views / note / depth_scale_m_per_unit 等、
    既知キー以外を保持し、save 時に書き戻す（既存 JSON との往復で情報を失わない）。
    """
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = ""
    coeffs: list = field(default_factory=list)
    fps: "float | None" = None
    extra: dict = field(default_factory=dict)   # calib_rms_px / note 等の残りキーを保持

    @classmethod
    def load(cls, path):
        """JSON から読み込む。既知キー以外は extra に保持する。"""
        with open(path) as f:
            meta = json.load(f)
        extra = {k: v for k, v in meta.items() if k not in _KNOWN_KEYS}
        return cls(
            width=int(meta["width"]),
            height=int(meta["height"]),
            fx=float(meta["fx"]), fy=float(meta["fy"]),
            cx=float(meta["cx"]), cy=float(meta["cy"]),
            model=meta.get("model", ""),
            coeffs=list(meta.get("coeffs", [])),
            fps=meta.get("fps", None),
            extra=extra,
        )

    def save(self, path):
        """JSON へ書き出す。extra のキーも書き戻す。"""
        meta = {
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx), "fy": float(self.fy),
            "cx": float(self.cx), "cy": float(self.cy),
            "model": self.model,
            "coeffs": list(self.coeffs),
            "fps": self.fps,
        }
        meta.update(self.extra)
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @property
    def K(self):
        """3x3 のカメラ行列（float64）。"""
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1.0]], dtype=np.float64)

    def scaled_to(self, width, height):
        """解像度比例スケール（fx*sx, fy*sy, cx*sx, cy*sy）した新しい Intrinsics を返す。

        アスペクト比が 1% を超えてずれる場合は警告を出す（クロップ撮影の検知）。
        """
        sx, sy = width / self.width, height / self.height
        if abs(sx - sy) > 0.01:
            print(f"[WARN] アスペクト比が不一致（intrinsics {self.width}x{self.height} "
                  f"vs 動画 {width}x{height}）。"
                  f"クロップ撮影の可能性があり、スケール変換では正確に補正できません。"
                  f"同条件での再キャリブレーションを推奨。")
        extra = dict(self.extra)
        extra["note"] = extra.get("note", "") + \
            f" / {self.width}x{self.height}から{width}x{height}へ比例スケール済み"
        return Intrinsics(
            width=int(width), height=int(height),
            fx=self.fx * sx, fy=self.fy * sy,
            cx=self.cx * sx, cy=self.cy * sy,
            model=self.model, coeffs=list(self.coeffs), fps=self.fps,
            extra=extra,
        )

    def needs_rectify(self, tol_px=1.0, tol_f=1e-3):
        """歪み係数・主点オフセット・fx/fy差のいずれかが無視できなければ True。"""
        coeffs = np.asarray(self.coeffs, dtype=float)
        if coeffs.size and np.abs(coeffs).max() > 1e-6:
            return True
        if abs(self.cx - self.width / 2) > tol_px or abs(self.cy - self.height / 2) > tol_px:
            return True
        f_mean = (self.fx + self.fy) / 2
        return abs(self.fx - self.fy) / f_mean > tol_f

    def rectified(self):
        """理想ピンホール化した Intrinsics（f=(fx+fy)/2, 主点=画像中央, 歪み0）を返す。"""
        f_new = (self.fx + self.fy) / 2
        dist = np.asarray(self.coeffs, dtype=float)
        extra = dict(self.extra)
        extra["note"] = extra.get("note", "") + " / rectify_video.py で理想ピンホール化済み"
        return Intrinsics(
            width=int(self.width), height=int(self.height),
            fx=f_new, fy=f_new, cx=self.width / 2, cy=self.height / 2,
            model="rectified_pinhole",
            coeffs=[0.0] * len(dist) if dist.size else [],
            fps=self.fps, extra=extra,
        )


def project(pts_cam, K):
    """pts_cam[...,3] (OpenCV系カメラ座標, m) を K で投影。

    Returns: (uv[...,2], valid[...])  valid は Z>1e-6
    """
    pts = np.asarray(pts_cam, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Z = pts[..., 2]
    valid = Z > 1e-6
    Zsafe = np.where(valid, Z, 1.0)
    u = fx * pts[..., 0] / Zsafe + cx
    v = fy * pts[..., 1] / Zsafe + cy
    uv = np.stack([u, v], axis=-1)
    return uv, valid


class VideoRectifier:
    """動画を理想ピンホール（主点=画像中央・fx=fy・歪みゼロ）へレクティファイする。

    旧 tools/rectify_video.py の rectify ロジックを移設（数値挙動を変えない）。
    """

    def __init__(self, intr):
        self.intr = intr

    def rectify_video(self, video_in, video_out):
        """理想ピンホール化して書き出す。Returns: (n_frames:int, intr_rect:Intrinsics)"""
        from handtraj.video_io import Mp4Writer

        intr = self.intr
        W, H = int(intr.width), int(intr.height)
        K = intr.K
        dist = np.asarray(intr.coeffs, dtype=float)
        f_new = (intr.fx + intr.fy) / 2
        K_new = np.array([[f_new, 0, W / 2], [0, f_new, H / 2], [0, 0, 1.0]])

        cap = cv2.VideoCapture(video_in)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, (W, H), cv2.CV_16SC2)
        writer = Mp4Writer(video_out, fps, (W, H))
        n = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break
            writer.write(cv2.remap(img, map1, map2, cv2.INTER_LINEAR))
            n += 1
        cap.release()
        writer.close()

        intr_rect = intr.rectified()
        shift = np.hypot(intr.cx - W / 2, intr.cy - H / 2)
        print(f"[INFO] レクティフィケーション完了: {n} frames -> {video_out}")
        print(f"[INFO] 主点シフト {shift:.1f}px を補正、f={f_new:.1f} "
              f"(fx={intr.fx:.1f}, fy={intr.fy:.1f})")
        return n, intr_rect


class CameraCalibrator:
    """チェスボードキャリブレーション（旧 calibrate_camera.py のロジックを移設）。

    pattern は「交点（内点）の数」 (cols, rows)。None なら AUTO_PATTERNS から自動検出する。
    square_mm は 1 マスの一辺 [mm]（K の推定値には影響しない・外部パラメータの物理尺度のみ）。
    """

    # パターン自動検出の候補（交点数 cols x rows）。先頭ほど優先。
    # (5,4)=6x5マス（手持ちボード）, (8,5)=9x6マス, (9,6)/(7,6) は市販キャリブボードに多い内点数。
    AUTO_PATTERNS = [(5, 4), (8, 5), (9, 6), (7, 6), (6, 5), (4, 5), (5, 8), (6, 9), (6, 7), (5, 6)]

    def __init__(self, pattern=None, square_mm=27.0):
        self.pattern = pattern
        self.square_mm = square_mm

    def detect_pattern(self, images, n_probe=6):
        """サンプル画像で候補パターンを総当たりし、最も検出率の高い (cols,rows) を返す。"""
        candidates = self.AUTO_PATTERNS
        probe = images[:: max(1, len(images) // n_probe)][:n_probe]
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        best, best_hits = None, 0
        for pat in candidates:
            hits = 0
            for img in probe:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                found, _ = cv2.findChessboardCorners(gray, pat, flags)
                hits += int(found)
            if hits > best_hits:
                best, best_hits = pat, hits
            if hits == len(probe):   # 全探査画像で検出できたら確定
                break
        if best is None:
            raise RuntimeError(
                "チェスボードを検出できません。盤全体が写りピントが合っているか、"
                "候補パターン外の盤（交点数を --cols/--rows で明示）でないか確認してください。")
        print(f"[INFO] パターン自動検出: 交点 {best[0]}x{best[1]} "
              f"(= {best[0]+1}x{best[1]+1} マス, 検出 {best_hits}/{len(probe)} 枚)")
        return best

    def _collect_corners(self, images, pattern, show_progress=True):
        """画像列からチェスボードコーナーを検出して (objpoints, imgpoints, size) を返す。"""
        objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
        objpoints, imgpoints = [], []
        size = None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(
                gray, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not found:
                continue
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp.copy())   # 同一参照だと後段のスケール処理が多重適用される
            imgpoints.append(corners)
            if show_progress:
                print(f"  [{len(imgpoints)}] frame {i}: corners OK")
        return objpoints, imgpoints, size

    def _calibrate(self, objpoints, imgpoints, size, square_mm):
        """objpoints を物理スケールにして cv2.calibrateCamera を回す。"""
        for p in objpoints:
            p[:, :2] *= square_mm / 1000.0   # 物理スケール（K には影響しないが再投影誤差表示のため）
        rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
        return rms, K, dist

    def _make_intrinsics(self, K, dist, size, fps, rms, n_views):
        """推定結果から Intrinsics を組み立てる（rms/views/note は extra へ）。"""
        intr = Intrinsics(
            width=int(size[0]), height=int(size[1]),
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
            model="opencv_pinhole",
            coeffs=[float(x) for x in np.ravel(dist)],
            fps=fps,
            extra={
                "calib_rms_px": float(rms),
                "calib_views": int(n_views),
                "note": "calibrate_camera.py によるチェスボードキャリブレーション結果。",
            },
        )
        print(f"[INFO] intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} "
              f"cx={intr.cx:.1f} cy={intr.cy:.1f} rms={rms:.3f}px views={n_views}")
        return intr

    def detect_pattern_or_self(self, images):
        """self.pattern があればそれを、無ければ自動検出した交点パターンを返す。"""
        return self.pattern if self.pattern else self.detect_pattern(images)

    def calibrate_images(self, images, fps=None, show_progress=True):
        """画像列からキャリブレーションして Intrinsics を返す。

        内部で _collect_corners 相当 + cv2.calibrateCamera を実行。
        rms は extra["calib_rms_px"], views は extra["calib_views"] に格納する。
        """
        pattern = self.detect_pattern_or_self(images)
        obj, imgp, size = self._collect_corners(images, pattern, show_progress=show_progress)
        rms, K, dist = self._calibrate(obj, imgp, size, self.square_mm)
        return self._make_intrinsics(K, dist, size, fps, rms, len(imgp))

    def calibrate_video(self, video_path, max_frames=30):
        """動画からキャリブレーションして Intrinsics を返す。"""
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        step = max(1, n // (max_frames * 2))   # 検出失敗を見込み2倍候補を等間隔サンプル
        imgs = []
        for t in range(0, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ret, img = cap.read()
            if ret:
                imgs.append(img)
        cap.release()
        fps = int(round(fps)) if fps else None

        pattern = self.detect_pattern_or_self(imgs)
        print(f"[INFO] 候補画像 {len(imgs)} 枚からコーナー検出（交点 {pattern[0]}x{pattern[1]}）...")
        obj, imgp, size = self._collect_corners(imgs, pattern)
        if len(imgp) < 10:
            raise RuntimeError(
                f"コーナー検出が {len(imgp)} 枚のみ（10枚以上必要）。"
                f"--cols/--rows（内点数=交点数）や撮影条件を確認してください。")
        obj, imgp = obj[:max_frames], imgp[:max_frames]
        rms, K, dist = self._calibrate(obj, imgp, size, self.square_mm)
        return self._make_intrinsics(K, dist, size, fps, rms, len(imgp))
