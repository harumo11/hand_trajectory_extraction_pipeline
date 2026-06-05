#!/usr/bin/env python3
"""
export.py — エクスポート段: world_joints × α → world_trajectory.npz（CLAUDE.md §3, §6-3）

本版（RGB 単眼）は α=1.0 既定（HaWoR 内蔵の Metric3D スケールを信頼）。
×α の機構は温存しており、将来のスケール較正は α の差し替えだけで実寸化できる:
  - 旧 D405 面-面較正（付録A）: m3_report=<m3_surface_report.json>
  - 手動/他方式の較正値:        alpha=<値>

出力 world_trajectory.npz:
  joints [T,2,21,3] (m)  … [left, right]、invalid な手は NaN
  alpha  スカラ
  valid  [T] bool（どちらかの手が valid）
  valid_per_hand [T,2] bool

診断: 骨長の解剖学的妥当性チェック（手長の目安 16〜20cm、最終関門の判定材料）と、
     m3_report 使用時は alpha_per_frame.png（時間ドリフト診断）。
"""
import json
import os

import numpy as np

from handtraj.hawor_adapter import MIDDLE_MCP, MIDDLE_TIP, WRIST


def hand_length_stats(joints, valid):
    """手長（手首→中指MCP→中指先端の経路長）の中央値 [m] を手ごとに返す。"""
    out = {}
    for h, name in enumerate(("left", "right")):
        j = joints[valid[:, h], h]            # [N,21,3]
        if len(j) == 0:
            out[name] = None
            continue
        L = (np.linalg.norm(j[:, MIDDLE_MCP] - j[:, WRIST], axis=1)
             + np.linalg.norm(j[:, MIDDLE_TIP] - j[:, MIDDLE_MCP], axis=1))
        out[name] = float(np.median(L))
    return out


def _plot_alpha_per_frame(per_frame, alpha_global, out_png, fps=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] plot skip: {e}")
        return
    idx = np.array([p["idx"] for p in per_frame])
    a = np.array([p["alpha"] for p in per_frame])
    x = idx / fps if fps else idx
    plt.figure(figsize=(9, 4))
    plt.plot(x, a, ".-", ms=3, lw=0.7, label="alpha_per_frame")
    plt.axhline(alpha_global, color="k", ls=":", label=f"alpha_global={alpha_global:.4f}")
    plt.xlabel("time [s]" if fps else "frame idx")
    plt.ylabel("alpha")
    plt.legend()
    plt.title("alpha per frame (drift diagnosis)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"[INFO] plot: {out_png}")


class TrajectoryExporter:
    """HaworSequence の world 手関節 × α を world_trajectory.npz として書き出す。"""

    def __init__(self, seq):
        self.seq = seq

    def export(self, out_npz, alpha=1.0, m3_report=None, fps=None):
        """world_joints × α → npz 保存（keys: joints/alpha/valid/valid_per_hand）。

        m3_report 指定時は alpha_global を優先し alpha_per_frame.png を出力。
        alpha は明示的な値を指定すると「指定値で適用」、None なら既定 1.0 として扱う
        （CLI の --alpha 省略時と一致する三分岐）。
        Returns: {"alpha":..., "T":..., "n_valid":..., "hand_length_m": {"left":..,"right":..}}
        手長チェックの [CHECK] print も行う。
        """
        per_frame = []
        if m3_report:
            with open(m3_report) as f:
                report = json.load(f)
            if report.get("status") != "ok":
                print(f"[ERROR] M3 レポートが不正: status={report.get('status')}")
                raise ValueError(f"M3 レポートが不正: status={report.get('status')}")
            alpha = float(report["alpha_global"])
            per_frame = report.get("alpha_per_frame", [])
            print(f"[INFO] α を M3 レポートから採用: {alpha:.4f}")
        elif alpha is not None:
            alpha = float(alpha)
            print(f"[INFO] α を指定値で適用: {alpha:.4f}")
        else:
            alpha = 1.0
            print("[INFO] α=1.0（HaWoR 内蔵 Metric3D スケールを信頼。誤差目安は CLAUDE.md §7）")

        joints, valid = self.seq.world_joints()   # [T,2,21,3] / [T,2]

        # ---- エクスポート本体: 掛け算のみ ----
        joints_true = joints * alpha

        out_dir = os.path.dirname(os.path.abspath(out_npz))
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            out_npz,
            joints=joints_true.astype(np.float32),
            alpha=np.float64(alpha),
            valid=valid.any(axis=1),
            valid_per_hand=valid,
        )
        T = joints_true.shape[0]
        n_valid = int(valid.any(axis=1).sum())
        print(f"[INFO] world_trajectory.npz: T={T}, alpha={alpha:.4f}, "
              f"valid={n_valid}/{T} -> {out_npz}")

        # ---- 診断: 手長の解剖学的妥当性（最終関門の判定材料）----
        stats = hand_length_stats(joints_true, valid)
        for name, L in stats.items():
            if L is None:
                print(f"[CHECK] {name}: valid フレームなし")
                continue
            cm = L * 100
            verdict = "妥当" if 14.0 <= cm <= 24.0 else "要確認（スケール誤差の可能性）"
            print(f"[CHECK] {name} 手長（手首→中指先端 経路長）中央値 = {cm:.1f} cm … {verdict}")

        if per_frame:
            _plot_alpha_per_frame(per_frame, alpha,
                                  os.path.join(out_dir, "alpha_per_frame.png"), fps=fps)

        return {"alpha": alpha, "T": T, "n_valid": n_valid, "hand_length_m": stats}
