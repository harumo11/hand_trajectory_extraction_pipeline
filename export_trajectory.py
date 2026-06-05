#!/usr/bin/env python3
"""
export_trajectory.py — エクスポート段の CLI シンウラッパ（CLAUDE.md §3, §6-3）

本体ロジックは handtraj.export.TrajectoryExporter / handtraj.hawor_adapter.HaworSequence へ移設済み。
本ファイルは引数解釈と [INFO]/[CHECK] 出力の互換維持のための薄い CLI ラッパ。

本版（RGB 単眼）は α=1.0 既定（HaWoR 内蔵の Metric3D スケールを信頼）。
×α の機構は温存しており、将来のスケール較正は α の差し替えだけで実寸化できる:
  - 旧 D405 面-面較正（付録A）: --m3_report <m3_surface_report.json>
  - 手動/他方式の較正値:        --alpha <値>

出力 world_trajectory.npz:
  joints [T,2,21,3] (m)  … [left, right]、invalid な手は NaN
  alpha  スカラ
  valid  [T] bool（どちらかの手が valid）
  valid_per_hand [T,2] bool

診断: 骨長の解剖学的妥当性チェック（手長の目安 16〜20cm、最終関門の判定材料）と、
     --m3_report 使用時は alpha_per_frame.png（時間ドリフト診断）。

使い方:
  python export_trajectory.py --hawor <take>/rgb --out <take>/world_trajectory.npz [--fps 30]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handtraj.hawor_adapter import HaworSequence
from handtraj.export import TrajectoryExporter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hawor", required=True, help="HaWoR seq_folder")
    ap.add_argument("--out", required=True, help="出力 world_trajectory.npz")
    ap.add_argument("--alpha", type=float, default=None,
                    help="グローバルスケール（既定 1.0。--m3_report 指定時はそちらを優先）")
    ap.add_argument("--m3_report", default=None,
                    help="旧 D405 面-面較正の m3_surface_report.json（付録A・任意）")
    ap.add_argument("--fps", type=float, default=None, help="診断プロットの時間軸用")
    args = ap.parse_args()

    seq = HaworSequence(args.hawor)
    try:
        TrajectoryExporter(seq).export(args.out, alpha=args.alpha,
                                       m3_report=args.m3_report, fps=args.fps)
    except ValueError:
        # M3 レポート不正時（[ERROR] は exporter 内で print 済み）
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
