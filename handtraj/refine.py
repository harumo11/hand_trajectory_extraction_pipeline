#!/usr/bin/env python3
"""
refine.py — 2D キーポイントによる手並進の事後リファイン（CLAUDE.md §7 の精度改善・HaWoR 無改造）。

背景:
  HaWoR の world 出力は画像面で ~25〜40px（手の距離 0.5m で約 1.2〜2cm）の整合誤差を持つ
  （支配要因: 並進回帰誤差・infiller の時間平滑化）。独立した 2D 検出器（MediaPipe、
  tools/detect_keypoints_2d.py で隔離 venv 実行）を基準に、フレーム×手ごとの
  **並進補正 Δt（カメラ座標系）のみ**を一括最適化して補正する。
  姿勢・形状は HaWoR のまま（剛体シフトなので骨長・関節角は不変）。

定式化:
  min_Δt  Σ_{t,h∈使用} Σ_j Huber( π_K(J_cam[t,h,j] + Δt[t,h]) − k2d[t,h,j] )
          + λ_prior‖Δt/σ_p‖² + λ_smooth‖(Δt[t+1]−Δt[t])/σ_v‖²
  - 深度方向は 21 点の透視広がりが拘束する（MANO 手サイズ固定のため可観測）
  - 未検出フレームは平滑項による補間
  - world への反映: J_world' = J_world + R_c2w·Δt

ロバスト化（実データ調査で必須と判明）:
  - 左右ラベル取り違え検知: 初期投影手首との距離で swap 判定
  - ゲート: 初期残差中央値 > gate_px の検出は破棄（誤検出対策）
  - Huber 損失（関節単位の外れ値対策）

自己検証（関門④a）:
  python -m handtraj.refine --selftest
"""
import argparse
import json
import os
import sys

import numpy as np

# OpenPose 手関節順の手首 index
WRIST = 0


# ============ 観測の読込・マッチング・ゲーティング ============
class KeypointObservations:
    """tools/detect_keypoints_2d.py の npz を読み込み、HaWoR 初期推定と突き合わせて
    「使用してよい観測」を決める（左右スワップ補正・誤検出ゲート込み）。"""

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.k2d = d["k2d"].astype(np.float64)        # [T,2,21,2] NaN=未検出
        self.detected = d["detected"].astype(bool)    # [T,2]
        self.score = d["score"].astype(np.float64)    # [T,2]
        self.T = int(d["n_frames"])
        self.stats = {}

    def match_and_gate(self, uv_init, valid, gate_px=80.0):
        """初期投影 uv_init [T,2,21,2] と照合し、使用マスク use[T,2] を返す（in-place で k2d を並べ替え）。

        1. 両手検出フレームで、(そのまま / 左右スワップ) の手首距離合計が小さい方を採用
        2. 手ごとに残差中央値 > gate_px の観測を破棄
        """
        T = min(self.T, uv_init.shape[0])
        k2d, det = self.k2d[:T], self.detected[:T]
        n_swap = 0
        for t in range(T):
            if det[t].all() and valid[t].all():
                d_keep = sum(np.linalg.norm(uv_init[t, h, WRIST] - k2d[t, h, WRIST]) for h in (0, 1))
                d_swap = sum(np.linalg.norm(uv_init[t, h, WRIST] - k2d[t, 1 - h, WRIST]) for h in (0, 1))
                if d_swap < d_keep:
                    k2d[t] = k2d[t, ::-1]
                    self.score[t] = self.score[t, ::-1]
                    n_swap += 1

        # 残差中央値ゲート（未検出スロットの All-NaN 警告を避けるため検出箇所のみ計算）
        res = np.linalg.norm(uv_init[:T] - k2d, axis=-1)          # [T,2,21]
        med = np.full((T, 2), np.inf)
        med[det] = np.nanmedian(res[det], axis=-1)
        use = det & valid[:T] & np.isfinite(med) & (med <= gate_px)
        self.k2d = k2d
        self.stats = {
            "n_frames": T,
            "n_detected": int(det.sum()),
            "n_swapped": n_swap,
            "n_gated_out": int((det & valid[:T]).sum() - use.sum()),
            "n_used": int(use.sum()),
            "init_residual_median_px": float(np.nanmedian(med[use])) if use.any() else None,
        }
        return use


# ============ 並進リファイナ ============
class RefineResult:
    """TranslationRefiner.solve の結果（補正量と診断情報）。"""

    def __init__(self, delta_t_cam, delta_t_world, use, diagnostics):
        self.delta_t_cam = delta_t_cam        # [T,2,3] m（カメラ座標系）
        self.delta_t_world = delta_t_world    # [T,2,3] m（world 座標系）
        self.use = use                        # [T,2] 最適化に使った観測
        self.diagnostics = diagnostics        # dict（残差統計など）

    def apply_world(self, joints_world):
        """world 関節へ補正を適用（剛体シフト）。invalid(NaN) はそのまま。"""
        return joints_world + self.delta_t_world[:, :, None, :]


class TranslationRefiner:
    """フレーム×手ごとの並進補正 Δt を torch で一括最適化する。"""

    def __init__(self, joints_cam, valid, K, R_c2w):
        """joints_cam [T,2,21,3] / valid [T,2] / K 3x3 / R_c2w [T,3,3]"""
        self.joints_cam = np.asarray(joints_cam, np.float64)
        self.valid = np.asarray(valid, bool)
        self.K = np.asarray(K, np.float64)
        self.R_c2w = np.asarray(R_c2w, np.float64)

    def project(self, pts):
        """pts [...,3] → uv [...,2]（numpy）"""
        z = pts[..., 2]
        return np.stack([self.K[0, 0] * pts[..., 0] / z + self.K[0, 2],
                         self.K[1, 1] * pts[..., 1] / z + self.K[1, 2]], -1)

    def solve(self, obs: KeypointObservations, gate_px=80.0,
              sigma_prior_m=0.10, sigma_vel_m=0.01, huber_px=10.0,
              iters=1500, lr=5e-4, fit_mask=None, verbose=True):
        """最適化を実行して RefineResult を返す。

        fit_mask [T] bool: 指定時、True のフレームの観測だけを当てはめに使う
        （ホールドアウト評価用。平滑項は全フレームに掛かる）。
        """
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        uv_init = self.project(self.joints_cam)                   # [T,2,21,2]
        use = obs.match_and_gate(uv_init, self.valid, gate_px)    # [T,2]
        T = use.shape[0]
        fit_use = use.copy()
        if fit_mask is not None:
            fit_use &= np.asarray(fit_mask, bool)[:T, None]

        J = torch.tensor(self.joints_cam[:T], device=dev)         # [T,2,21,3]
        k2d = torch.tensor(np.nan_to_num(obs.k2d[:T]), device=dev)
        m = torch.tensor(fit_use, device=dev)                     # [T,2]
        K = torch.tensor(self.K, device=dev)
        dt = torch.zeros(T, 2, 3, device=dev, requires_grad=True)
        opt = torch.optim.Adam([dt], lr=lr)
        huber = torch.nn.HuberLoss(reduction="none", delta=huber_px)

        for it in range(iters):
            opt.zero_grad()
            P = J + dt[:, :, None, :]
            z = P[..., 2].clamp(min=1e-3)
            uv = torch.stack([K[0, 0] * P[..., 0] / z + K[0, 2],
                              K[1, 1] * P[..., 1] / z + K[1, 2]], -1)
            r = huber(uv, k2d).sum(-1)                            # [T,2,21]
            loss_data = (r.mean(-1) * m).sum() / m.sum().clamp(min=1)
            loss_prior = ((dt / sigma_prior_m) ** 2).mean()
            loss_smooth = (((dt[1:] - dt[:-1]) / sigma_vel_m) ** 2).mean()
            loss = loss_data + loss_prior + loss_smooth
            loss.backward()
            opt.step()
            if verbose and (it + 1) % 200 == 0:
                print(f"[refine] iter {it+1}/{iters}  data={loss_data.item():.2f} "
                      f"prior={loss_prior.item():.4f} smooth={loss_smooth.item():.4f}")

        dt_np = dt.detach().cpu().numpy()
        dt_world = np.einsum("tij,thj->thi", self.R_c2w[:T], dt_np)

        # 診断: 使用観測に対する残差 before/after（中央値）
        uv_after = self.project(self.joints_cam[:T] + dt_np[:, :, None, :])
        res_b = np.linalg.norm(uv_init[:T] - obs.k2d[:T], axis=-1)
        res_a = np.linalg.norm(uv_after - obs.k2d[:T], axis=-1)
        diag = {
            **obs.stats,
            "residual_before_px": float(np.nanmedian(res_b[use])),
            "residual_after_px": float(np.nanmedian(res_a[use])),
            "delta_t_norm_median_m": float(np.median(np.linalg.norm(dt_np[use], axis=-1))),
            "delta_t_norm_max_m": float(np.max(np.linalg.norm(dt_np[use], axis=-1))) if use.any() else 0.0,
        }
        if fit_mask is not None:
            ho = use & ~fit_use      # 当てはめに使わなかった観測（ホールドアウト）
            if ho.any():
                diag["holdout_before_px"] = float(np.nanmedian(res_b[ho]))
                diag["holdout_after_px"] = float(np.nanmedian(res_a[ho]))
        return RefineResult(dt_np, dt_world, use, diag)


def refine_npz(base_npz, result: RefineResult, out_npz=None, report_json=None):
    """world_trajectory.npz に補正を適用して書き戻す（後方互換: 既存キー不変+追加キー）。"""
    d = dict(np.load(base_npz))
    T = min(d["joints"].shape[0], result.delta_t_world.shape[0])
    joints = d["joints"].astype(np.float64)
    joints[:T] += result.delta_t_world[:T, :, None, :] * float(d["alpha"])
    d["joints"] = joints.astype(np.float32)
    d["refined"] = np.bool_(True)
    d["delta_t_world"] = result.delta_t_world.astype(np.float32)
    np.savez(out_npz or base_npz, **d)
    if report_json:
        with open(report_json, "w") as f:
            json.dump(result.diagnostics, f, indent=2, ensure_ascii=False)
    return out_npz or base_npz


# ============ 高水準 API（CLI / Pipeline 共用） ============
def refine_take(take_dir, keypoints=None, sigma_vel_m=0.003, gate_px=80.0,
                skip_overlay=False):
    """テイクディレクトリ一式に対してリファインを実行する。

    前提: <take>/rgb/（HaWoR出力）, <take>/intrinsics.json, <take>/world_trajectory.npz,
         <take>/keypoints_2d.npz（tools/detect_keypoints_2d.py の出力）
    出力: world_trajectory.npz を更新（refined=True, delta_t_world 追加）
         + refine_report.json + overlay_refined/（skip_overlay=False 時）
    Returns: 診断 dict
    """
    from .hawor_adapter import HaworSequence
    from .camera import Intrinsics

    take = os.path.abspath(take_dir)
    seq_folder = os.path.join(take, "rgb")
    kp_path = keypoints or os.path.join(take, "keypoints_2d.npz")
    base_npz = os.path.join(take, "world_trajectory.npz")
    for p, hint in [(kp_path, "tools/detect_keypoints_2d.py（隔離 venv）で検出が必要"),
                    (base_npz, "先に run_pipeline でエクスポートが必要"),
                    (os.path.join(take, "intrinsics.json"), "intrinsics.json が必要")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} がありません（{hint}）")
    if bool(np.load(base_npz).get("refined", False)):
        raise RuntimeError(f"{base_npz} は既にリファイン済み（二重適用防止）。"
                           f"run_pipeline --force で base を作り直してから再実行してください。")

    seq = HaworSequence(seq_folder)
    intr = Intrinsics.load(os.path.join(take, "intrinsics.json"))
    joints_cam = (np.einsum("tij,thnj->thni", seq.R_w2c, seq.joints_world)
                  + seq.t_w2c[:, None, None, :])

    obs = KeypointObservations(kp_path)
    res = TranslationRefiner(joints_cam, seq.valid, intr.K, seq.R_c2w).solve(
        obs, gate_px=gate_px, sigma_vel_m=sigma_vel_m)
    refine_npz(base_npz, res, report_json=os.path.join(take, "refine_report.json"))
    d = res.diagnostics
    print(f"[INFO] リファイン適用: 残差 {d['residual_before_px']:.1f}px -> {d['residual_after_px']:.1f}px "
          f"(使用 {d['n_used']}観測, swap {d['n_swapped']}, gate除外 {d['n_gated_out']})")
    print(f"[INFO] |Δt| 中央値 {d['delta_t_norm_median_m']*100:.2f}cm / 最大 {d['delta_t_norm_max_m']*100:.2f}cm")
    print(f"[INFO] 更新: {base_npz}（refined=True, delta_t_world 追加）")

    if not skip_overlay:
        from .visualization import OverlayRenderer

        class _RefinedView:
            """Δt 適用後の verts_cam を見せる OverlayRenderer 用ビュー。"""
            def __init__(self, seq, dt):
                T = min(seq.T, dt.shape[0])
                self.verts_cam = seq.verts_cam.copy()
                self.verts_cam[:T] += dt[:T, :, None, :]
                self.valid = seq.valid

        out_dir = os.path.join(take, "overlay_refined")
        renderer = OverlayRenderer(_RefinedView(seq, res.delta_t_cam), intr)
        renderer.snapshots(os.path.join(take, "rgb.mp4"), out_dir, num=6)
        renderer.render_video(os.path.join(take, "rgb.mp4"), os.path.join(out_dir, "overlay.mp4"))
        print(f"[INFO] リファイン後オーバーレイ: {out_dir}/（補正前は overlay/ と比較）")
    return d


# ============ selftest（関門④a）: 人工オフセットの復元 ============
def _make_synthetic(T=200, seed=0):
    """簡易な合成シーン: 2本の手（21点クラスタ）が動き、真の Δt(ランダムウォーク±2cm)でずれた観測を作る。"""
    rng = np.random.default_rng(seed)
    K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
    # 手モデル: 手首原点 + 広がり ~8cm の 21 点（固定形状）
    local = rng.uniform(-0.04, 0.08, (21, 3))
    local[WRIST] = 0
    # 手首軌道（ゆっくり移動、Z 0.4-0.6m）
    def traj(phase):
        s = np.linspace(0, 2 * np.pi, T)
        return np.stack([0.12 * np.sin(s + phase), 0.08 * np.cos(1.3 * s + phase),
                         0.5 + 0.08 * np.sin(0.7 * s + phase)], -1)
    J = np.stack([traj(0), traj(2.0)], 1)[:, :, None, :] + local[None, None]   # [T,2,21,3]
    valid = np.ones((T, 2), bool)
    # 真の並進誤差: 平滑ランダムウォーク（std ~1cm, max ~2cm 程度）
    steps = rng.normal(0, 0.0015, (T, 2, 3))
    dt_true = np.cumsum(steps, axis=0)
    dt_true -= dt_true.mean(axis=0, keepdims=True)
    dt_true = np.clip(dt_true, -0.02, 0.02)
    # 観測 = 真位置(J+dt_true)の投影 + 画素ノイズ。1割は未検出に
    z = (J + dt_true[:, :, None, :])[..., 2]
    uv = np.stack([K[0, 0] * (J + dt_true[:, :, None, :])[..., 0] / z + K[0, 2],
                   K[1, 1] * (J + dt_true[:, :, None, :])[..., 1] / z + K[1, 2]], -1)
    uv += rng.normal(0, 3.0, uv.shape)
    detected = rng.random((T, 2)) > 0.1
    uv[~detected] = np.nan
    return J, valid, K, dt_true, uv, detected


def run_selftest():
    import tempfile
    J, valid, K, dt_true, uv, detected = _make_synthetic()
    T = J.shape[0]
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f.name, k2d=uv.astype(np.float32), detected=detected,
                 score=detected.astype(np.float32), width=1280, height=720, n_frames=T)
        obs = KeypointObservations(f.name)
    os.unlink(f.name)

    refiner = TranslationRefiner(J, valid, K, np.tile(np.eye(3), (T, 1, 1)))
    res = refiner.solve(obs, verbose=False)

    # 復元率は面内(x,y)と奥行き(z)で分けて評価する。
    # 奥行きは「21点の透視広がりの変化」からしか観測できず、画素ノイズ3px下では
    # 数mm の不確かさが情報限界として残る（面内は f/Z ≈ 2000px/m で強く拘束される）。
    err = res.delta_t_cam - dt_true
    rec_ip = 1.0 - (np.linalg.norm(err[..., :2], axis=-1).mean()
                    / np.linalg.norm(dt_true[..., :2], axis=-1).mean())
    rec_z = 1.0 - np.abs(err[..., 2]).mean() / np.abs(dt_true[..., 2]).mean()
    rb, ra = res.diagnostics["residual_before_px"], res.diagnostics["residual_after_px"]
    print(f"[selftest] 面内(x,y) 復元率   = {rec_ip*100:.1f}%  (基準 ≥90%)")
    print(f"[selftest] 奥行き(z) 復元率   = {rec_z*100:.1f}%  (情報限界により低め・改善していればOK)")
    print(f"[selftest] 再投影残差          = {rb:.1f}px -> {ra:.1f}px  (基準 ≤4.5px=注入ノイズ3pxの1.5倍)")
    ok = rec_ip >= 0.90 and rec_z > 0.0 and ra <= 4.5
    print(f"[selftest] 関門④a: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return run_selftest()
    ap.error("--selftest を指定（実データは scripts/refine_trajectory.py から）")


if __name__ == "__main__":
    sys.exit(main())
