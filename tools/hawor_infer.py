#!/usr/bin/env python3
"""
hawor_infer.py — HaWoR 推論ドライバ（demo.py と同一の処理列、可視化なし）。

demo.py は冒頭で可視化系（pytorch3d renderer / aitviewer）を import するため、
ここでは demo.py が呼ぶ 4 つの内部関数だけを同じ順序で呼ぶ（HaWoR 本体は無改造）:
  1. detect_track_video       … フレーム抽出 + 手検出/トラッキング
  2. hawor_motion_estimation  … 手 MANO 推定（カメラ座標）+ 手マスク生成
  3. hawor_slam               … masked DROID-SLAM + Metric3D スケール → カメラ軌道
  4. hawor_infiller           … 欠損補間 + world 合成 → world_space_res.pth

注意:
  - HaWoR 内部は cwd 相対パス（weights/ や _DATA/）前提のため、HaWoR ルートを cwd にして実行する。
  - --img_focal は必ず実カメラの fx を渡すこと（省略時 HaWoR は 600 にフォールバックする）。
    intrinsics.json があれば --intrinsics で自動読込できる。

使い方:
  python tools/hawor_infer.py --video_path /abs/path/captures/take01/rgb.mp4 \
                              --intrinsics /abs/path/captures/take01/intrinsics.json
  → seq_folder（<video_dir>/rgb/）に world_space_res.pth と SLAM/*.npz が生成される
"""
import argparse
import glob
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HAWOR_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "third_party", "HaWoR"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_path", required=True, help="入力動画（絶対パス推奨）")
    ap.add_argument("--intrinsics", default=None, help="intrinsics.json（fx を img_focal に使用）")
    ap.add_argument("--img_focal", type=float, default=None, help="fx [px]（--intrinsics より優先）")
    ap.add_argument("--checkpoint", default="./weights/hawor/checkpoints/hawor.ckpt")
    ap.add_argument("--infiller_weight", default="./weights/hawor/checkpoints/infiller.pt")
    args = ap.parse_args()

    args.video_path = os.path.abspath(args.video_path)
    args.input_type = "file"
    args.vis_mode = None

    if args.img_focal is None and args.intrinsics:
        with open(args.intrinsics) as f:
            meta = json.load(f)
        args.img_focal = float(meta["fx"])
        print(f"[INFO] img_focal = fx = {args.img_focal} (from {args.intrinsics})")
    if args.img_focal is None:
        print("[WARN] img_focal 未指定。HaWoR は 600px にフォールバックします（精度低下）。")

    # torch>=2.6 は torch.load の既定が weights_only=True となり、omegaconf 等を含む
    # HaWoR/Metric3D の公式 ckpt が読めない。重みは公式配布（信頼済みソース）のため、
    # このドライバプロセス内に限り旧挙動へ戻す（HaWoR 本体は無改造）。
    import torch
    _torch_load_orig = torch.load

    def _torch_load_compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return _torch_load_orig(*a, **kw)

    torch.load = _torch_load_compat

    # HaWoR 内部の相対パス前提に合わせる
    os.chdir(HAWOR_ROOT)
    sys.path.insert(0, HAWOR_ROOT)
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation, hawor_infiller
    from scripts.scripts_test_video.hawor_slam import hawor_slam

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    print(f"[INFO] frames: {start_idx}..{end_idx} ({len(imgfiles)} imgs), seq_folder={seq_folder}")

    res_path = os.path.join(seq_folder, "world_space_res.pth")
    if os.path.exists(res_path):
        print(f"[INFO] 既存の {res_path} があるためスキップ（再実行する場合は削除）")
        return 0

    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)
    assert os.path.exists(slam_path), f"SLAM 出力が生成されていません: {slam_path}"

    hawor_infiller(args, start_idx, end_idx, frame_chunks_all)  # world_space_res.pth を保存
    assert os.path.exists(res_path), f"world_space_res.pth が生成されていません: {res_path}"
    print(f"[INFO] HaWoR 推論完了: {res_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
