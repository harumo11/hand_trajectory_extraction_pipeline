#!/usr/bin/env python3
"""
calibrate_camera.py — チェスボードで RGB カメラの内部パラメータを推定し intrinsics.json を出力する。

HaWoR は焦点距離が未指定だと 600px 仮定にフォールバックして精度が落ちるため（CLAUDE.md §2）、
録画に使うカメラを一度キャリブレーションして intrinsics.json を作っておく。

使い方:
  # 6x5マス・27mm のボード（内点=交点は 5x4）をいろんな角度で撮った動画から:
  python calibrate_camera.py --video calib.mp4 --out intrinsics.json
  #   ↑ --cols/--rows 省略時はパターン自動検出（マス数/交点数の取り違えを吸収）
  # 明示指定する場合（交点の数で指定）:
  python calibrate_camera.py --video calib.mp4 --cols 5 --rows 4 --square_mm 27
  # 連番画像からも可:
  python calibrate_camera.py --images 'calib/*.jpg' --out intrinsics.json
  # 自己検証（カメラ不要・関門②'）: 既知 K の合成チェスボード画像から K を復元
  python calibrate_camera.py --selftest

注意:
  - --cols/--rows は「交点（内点）の数」。6x5 マスのボードなら交点は 5x4。
    省略すれば候補パターンを総当たりで自動判定するので迷ったら省略でよい。
  - --square_mm（1マスの一辺）は K の推定値には影響しない（外部パラメータの物理尺度のみ）。
  - 動画は等間隔サンプリングし、コーナー検出に成功した最大 --max_frames 枚を使用。
  - 出力形式は旧 split_bag.py の intrinsics.json と互換（depth_scale 無し）。

実装は handtraj.camera.CameraCalibrator に移設済み。本ファイルは argparse の薄いラッパ。
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handtraj.camera import CameraCalibrator  # noqa: E402


# ============ selftest: 合成チェスボードで K 復元（関門②'） ============
def render_chessboard(K, dist, rvec, tvec, pattern, square_m, size):
    """既知 K でチェスボード平面を投影描画した合成画像を返す。"""
    W, H = size
    cols, rows = pattern
    # 外周1マス分の余白を含む盤面の四隅を投影し、平面ホモグラフィで盤画像を貼る
    board_img = np.full(((rows + 3) * 60, (cols + 3) * 60), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0, x0 = (r + 1) * 60, (c + 1) * 60
                board_img[y0:y0 + 60, x0:x0 + 60] = 0
    # 盤画像の画素 ↔ 物理座標（交点(0,0) が board_img の (120,120)px に対応）
    src = np.float32([[120, 120], [120 + cols * 60, 120],
                      [120 + cols * 60, 120 + rows * 60], [120, 120 + rows * 60]])
    obj = np.float32([[0, 0, 0], [cols * square_m, 0, 0],
                      [cols * square_m, rows * square_m, 0], [0, rows * square_m, 0]])
    img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    Hmat = cv2.getPerspectiveTransform(src, img_pts.reshape(4, 2).astype(np.float32))
    return cv2.warpPerspective(board_img, Hmat, (W, H),
                               flags=cv2.INTER_LINEAR, borderValue=200)


def _selftest_case(label, pattern, square_m, auto_detect, seed=0):
    rng = np.random.default_rng(seed)
    W, H = 1280, 720
    K_true = np.array([[900.0, 0, W / 2 + 8], [0, 905.0, H / 2 - 5], [0, 0, 1]])
    dist_true = np.array([-0.05, 0.01, 0, 0, 0])

    imgs = []
    for _ in range(25):
        rvec = np.deg2rad(rng.uniform(-25, 25, 3))
        tvec = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.04, 0.04),
                         rng.uniform(0.35, 0.6)])
        tvec[:2] -= np.array([pattern[0], pattern[1]]) * square_m / 2  # 盤中心を画面中央付近へ
        imgs.append(render_chessboard(K_true, dist_true, rvec, tvec, pattern, square_m, (W, H)))

    # キャリブレーションは CameraCalibrator 経由で実行（明示指定 or 自動検出）。
    calib = CameraCalibrator(pattern=None if auto_detect else pattern, square_mm=square_m * 1000)
    used = calib.detect_pattern(imgs) if auto_detect else pattern
    assert used == pattern, f"自動検出が不一致: {used} != {pattern}"
    obj, imgp, size = calib._collect_corners(imgs, used, show_progress=False)
    print(f"[selftest:{label}] コーナー検出成功: {len(imgp)}/{len(imgs)} 枚")
    assert len(imgp) >= 10, "検出枚数不足"
    rms, K, dist = calib._calibrate(obj, imgp, size, square_m * 1000)
    err_fx = abs(K[0, 0] - K_true[0, 0]) / K_true[0, 0] * 100
    err_fy = abs(K[1, 1] - K_true[1, 1]) / K_true[1, 1] * 100
    print(f"[selftest:{label}] true fx={K_true[0,0]:.1f} -> 推定 {K[0,0]:.1f} (誤差 {err_fx:.2f}%)")
    print(f"[selftest:{label}] true fy={K_true[1,1]:.1f} -> 推定 {K[1,1]:.1f} (誤差 {err_fy:.2f}%)")
    print(f"[selftest:{label}] 再投影 RMS = {rms:.3f}px")
    return err_fx < 1.0 and err_fy < 1.0


def run_selftest():
    # case1: 9x6マス相当（交点8x5・25mm）を明示指定で
    ok1 = _selftest_case("8x5/25mm/明示", (8, 5), 0.025, auto_detect=False, seed=0)
    # case2: 手持ちボード 6x5マス（交点5x4・27mm）を自動検出経路で
    ok2 = _selftest_case("5x4/27mm/自動検出", (5, 4), 0.027, auto_detect=True, seed=1)
    ok = ok1 and ok2
    print(f"[selftest] 関門②': {'PASS' if ok else 'FAIL'} (両ケースで fx/fy 誤差 <1%)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--video", help="チェスボードを撮影した動画")
    ap.add_argument("--images", help="連番画像の glob パターン（例 'calib/*.jpg'）")
    ap.add_argument("--out", default="intrinsics.json")
    ap.add_argument("--cols", type=int, default=None,
                    help="チェスボード内点（交点）の列数。省略で自動検出")
    ap.add_argument("--rows", type=int, default=None,
                    help="チェスボード内点（交点）の行数。省略で自動検出")
    ap.add_argument("--square_mm", type=float, default=27.0,
                    help="1マスの一辺 [mm]（K には影響しない）")
    ap.add_argument("--max_frames", type=int, default=30, help="使用する最大フレーム数")
    args = ap.parse_args()

    if args.selftest:
        return run_selftest()
    if not args.video and not args.images:
        ap.error("--video か --images を指定してください（または --selftest）")
    if (args.cols is None) != (args.rows is None):
        ap.error("--cols と --rows は両方指定するか両方省略（自動検出）してください")

    pattern = (args.cols, args.rows) if args.cols else None
    calib = CameraCalibrator(pattern=pattern, square_mm=args.square_mm)

    if args.video:
        try:
            intr = calib.calibrate_video(args.video, max_frames=args.max_frames)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            return 1
    else:
        imgs = [cv2.imread(p) for p in sorted(glob.glob(args.images))]
        used = calib.detect_pattern_or_self(imgs)
        print(f"[INFO] 候補画像 {len(imgs)} 枚からコーナー検出（交点 {used[0]}x{used[1]}）...")
        obj, imgp, size = calib._collect_corners(imgs, used)
        if len(imgp) < 10:
            print(f"[ERROR] コーナー検出が {len(imgp)} 枚のみ（10枚以上必要）。"
                  f"--cols/--rows（内点数=交点数）や撮影条件を確認してください。")
            return 1
        obj, imgp = obj[:args.max_frames], imgp[:args.max_frames]
        rms, K, dist = calib._calibrate(obj, imgp, size, args.square_mm)
        intr = calib._make_intrinsics(K, dist, size, None, rms, len(imgp))

    intr.save(args.out)
    rms = intr.extra["calib_rms_px"]
    print(f"[INFO] 出力: {args.out}")
    if rms > 1.0:
        print(f"[WARN] 再投影 RMS が大きい ({rms:.2f}px)。ブレの少ない動画で撮り直し推奨。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
