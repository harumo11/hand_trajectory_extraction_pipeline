#!/usr/bin/env python3
"""
run_pipeline.py — RGB 単眼パイプライン: 取込 → HaWoR → 検証 → エクスポート（CLAUDE.md §6-4）

ステージ（生成物が既にあればスキップ。--force で再実行）:
  取込    : --video の mp4 を <take>/rgb.mp4 へ配置（--intrinsics も同様に配置・任意）
  補正    : intrinsics があれば理想ピンホール化（歪み補正+主点センタリング、--no_rectify で無効）。
            HaWoR の内部仮定（fx=fy・主点=中央・歪み0）と入力を一致させる衛生措置
            （注: 整合精度の実測改善は未確認 — CLAUDE.md §7。歪みの大きいカメラでは有効なはず）。
            元動画は <take>/rgb_raw.mp4、元 intrinsics は <take>/intrinsics_raw.json に保持。
  HaWoR  : tools/hawor_infer.py → <take>/rgb/{world_space_res.pth, SLAM/*.npz}
  検証    : OverlayRenderer → <take>/overlay/*.png + overlay.mp4（関門③'・目視確認）
            intrinsics 不在時は SLAM の推定焦点距離から擬似 intrinsics を生成（精度注意）
  エクスポート: TrajectoryExporter → <take>/world_trajectory.npz（既定 α=1.0）
  3D可視化 : Skeleton3DRenderer → <take>/vis3d/world_3d.mp4（カメラ軌跡+手スケルトン）

本ファイルは argparse → PipelineConfig → Pipeline().run() の薄いラッパ。
本体ロジックは handtraj.pipeline.Pipeline へ移設済み。

使い方:
  # 録画済み/任意の mp4 から
  python scripts/run_pipeline.py --video path/to/video.mp4 --take captures/take01 \
                         [--intrinsics intrinsics.json]
  # 取込済みディレクトリ（<take>/rgb.mp4 が存在）から
  python scripts/run_pipeline.py --take captures/take01
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))  # プロジェクトルート
from handtraj.pipeline import Pipeline, PipelineConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="入力 mp4（省略時は <take>/rgb.mp4 を使用）")
    ap.add_argument("--take", required=True, help="テイクディレクトリ（成果物の置き場）")
    ap.add_argument("--intrinsics", default=None,
                    help="calibrate_camera.py の intrinsics.json（任意・強く推奨）")
    ap.add_argument("--force", action="store_true", help="生成物があっても再実行")
    ap.add_argument("--skip_overlay", action="store_true", help="関門③'のオーバーレイ生成を省略")
    ap.add_argument("--skip_vis3d", action="store_true", help="3D 可視化動画の生成を省略")
    ap.add_argument("--no_rectify", action="store_true",
                    help="理想ピンホール化（歪み補正+主点センタリング）を行わない")
    ap.add_argument("--refine", action="store_true",
                    help="2D キーポイントによる並進リファインを実行"
                         "（要 .venv_mp: bash tools/setup_keypoint_env.sh で構築）")
    args = ap.parse_args()

    cfg = PipelineConfig(
        take=args.take,
        video=args.video,
        intrinsics=args.intrinsics,
        force=args.force,
        skip_overlay=args.skip_overlay,
        skip_vis3d=args.skip_vis3d,
        no_rectify=args.no_rectify,
        refine=args.refine,
    )
    return Pipeline(cfg).run()


if __name__ == "__main__":
    sys.exit(main())
