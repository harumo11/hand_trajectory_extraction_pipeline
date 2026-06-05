#!/usr/bin/env python3
"""
pipeline.py — RGB 単眼パイプライン本体（CLAUDE.md §6-4）。

旧 run_pipeline.py のステージ列を PipelineConfig + Pipeline へ移設したもの。
ステージ（生成物が既にあればスキップ。force で再実行）:
  取込    : video の mp4 を <take>/rgb.mp4 へ配置（intrinsics も同様に配置・任意）
  補正    : intrinsics があれば理想ピンホール化（歪み補正+主点センタリング、no_rectify で無効）。
            HaWoR の内部仮定（fx=fy・主点=中央・歪み0）と入力を一致させる衛生措置
            （注: 整合精度の実測改善は未確認 — CLAUDE.md §7。歪みの大きいカメラでは有効なはず）。
            元動画は <take>/rgb_raw.mp4、元 intrinsics は <take>/intrinsics_raw.json に保持。
  HaWoR  : tools/hawor_infer.py を subprocess で実行（プロセス分離は設計意図・維持）
            → <take>/rgb/{world_space_res.pth, SLAM/*.npz}
  検証    : OverlayRenderer → <take>/overlay/*.png + overlay.mp4（関門③'・目視確認）
            intrinsics 不在時は SLAM の推定焦点距離から擬似 intrinsics を生成（精度注意）
  エクスポート: TrajectoryExporter → <take>/world_trajectory.npz（既定 α=1.0）
  3D可視化 : Skeleton3DRenderer → <take>/vis3d/world_3d.mp4（カメラ軌跡+手スケルトン）

overlay/export/vis3d は subprocess ではなく handtraj のクラスを直接呼ぶ
（出力ファイル・スキップ判定・ログ文言は旧 run_pipeline.py と互換）。
HaWoR 推論のみ tools/hawor_infer.py を subprocess 実行する（[RUN] 行は subprocess 実行時のみ）。
"""
import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

from handtraj.camera import Intrinsics, VideoRectifier
from handtraj.export import TrajectoryExporter
from handtraj.hawor_adapter import HaworSequence
from handtraj.video_io import video_fps, video_size
from handtraj.visualization import OverlayRenderer, Skeleton3DRenderer

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PY = sys.executable


@dataclass
class PipelineConfig:
    """パイプラインの設定。CLI 引数（run_pipeline.py）と 1:1 対応する。"""
    take: str
    video: "str | None" = None
    intrinsics: "str | None" = None
    force: bool = False
    skip_overlay: bool = False
    skip_vis3d: bool = False
    no_rectify: bool = False


class Pipeline:
    """取込 → 補正 → HaWoR → 検証 → エクスポート → 3D 可視化 を順に実行する。

    各段のスキップ判定・ログ文言は旧 run_pipeline.py と同一挙動を保つ。
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.take = os.path.abspath(cfg.take)
        os.makedirs(self.take, exist_ok=True)
        self.rgb = os.path.join(self.take, "rgb.mp4")
        self.intr = os.path.join(self.take, "intrinsics.json")
        self.seq_folder = os.path.join(self.take, "rgb")   # HaWoR: <video_dir>/<basename>/
        self.res_pth = os.path.join(self.seq_folder, "world_space_res.pth")
        self.traj_npz = os.path.join(self.take, "world_trajectory.npz")
        self.has_intr = False

    # ---- subprocess ヘルパ（[RUN] 行は subprocess 実行時のみ）----
    def _run(self, cmd, **kw):
        print(f"\n[RUN] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, **kw)

    # ---- 取込 ----
    def ingest(self):
        """video/intrinsics を <take> へ配置し、intrinsics を動画解像度に合わせる。"""
        cfg = self.cfg
        if cfg.video:
            src = os.path.abspath(cfg.video)
            if src != self.rgb and (cfg.force or not os.path.exists(self.rgb)):
                shutil.copy2(src, self.rgb)
                print(f"[INFO] 取込: {src} -> {self.rgb}")
        assert os.path.exists(self.rgb), f"入力がありません: {self.rgb}（--video を指定）"
        if cfg.intrinsics:
            src = os.path.abspath(cfg.intrinsics)
            if src != self.intr:
                shutil.copy2(src, self.intr)
        self.has_intr = os.path.exists(self.intr)
        if self.has_intr:
            self._adjust_intrinsics_to_video()   # 解像度ミスマッチの安全装置

    def _adjust_intrinsics_to_video(self):
        """intrinsics の解像度と動画の解像度が違う場合、fx/fy/cx/cy を比例スケールして上書きする。
        （キャリブレーション解像度 ≠ 録画解像度の事故を防ぐ安全装置）"""
        intr = Intrinsics.load(self.intr)
        vw, vh = video_size(self.rgb)
        if (vw, vh) == (int(intr.width), int(intr.height)):
            return
        # scaled_to がアスペクト比不一致時の [WARN] と note 追記を担う
        scaled = intr.scaled_to(vw, vh)
        scaled.save(self.intr)
        print(f"[INFO] intrinsics を動画解像度 {vw}x{vh} に合わせてスケール（fx={scaled.fx:.1f}）")

    # ---- 補正: 理想ピンホール化（HaWoR の内部仮定と入力を一致させる）----
    def rectify(self):
        cfg = self.cfg
        rect_done = os.path.join(self.take, ".rectified")
        if self.has_intr and not cfg.no_rectify and (cfg.force or not os.path.exists(rect_done)):
            intr = Intrinsics.load(self.intr)
            if intr.needs_rectify():
                rgb_raw = os.path.join(self.take, "rgb_raw.mp4")
                intr_raw = os.path.join(self.take, "intrinsics_raw.json")
                if not os.path.exists(rgb_raw):
                    os.replace(self.rgb, rgb_raw)
                    shutil.copy2(self.intr, intr_raw)
                intr_src = Intrinsics.load(intr_raw)
                _, intr_rect = VideoRectifier(intr_src).rectify_video(rgb_raw, self.rgb)
                intr_rect.save(self.intr)
                print(f"[INFO] 新 intrinsics: {self.intr}")
            else:
                print("[INFO] 補正不要（既に理想ピンホール相当）")
            with open(rect_done, "w") as f:
                f.write("done\n")
        if not self.has_intr:
            print("[WARN] intrinsics.json なし。HaWoR は焦点距離 600px 仮定で動作（精度低下）。"
                  "calibrate_camera.py での実測を推奨。")

    # ---- HaWoR 推論（プロセス分離維持・subprocess）----
    def infer(self):
        cfg = self.cfg
        if cfg.force and os.path.exists(self.res_pth):
            os.remove(self.res_pth)
        if not os.path.exists(self.res_pth):
            cmd = [PY, os.path.join(ROOT, "tools", "hawor_infer.py"), "--video_path", self.rgb]
            if self.has_intr:
                cmd += ["--intrinsics", self.intr]
            self._run(cmd)
        assert os.path.exists(self.res_pth), f"HaWoR 出力がありません: {self.res_pth}"

    # ---- 関門③': 投影オーバーレイ ----
    def overlay(self):
        if self.cfg.skip_overlay:
            return
        ov_intr_path = self.intr if self.has_intr else self._make_pseudo_intrinsics(
            os.path.join(self.seq_folder, "intrinsics_pseudo.json"))
        intr = Intrinsics.load(ov_intr_path)
        seq = HaworSequence(self.seq_folder)
        if not seq.valid.any(axis=1).any():
            print("[ERROR] valid フレームがありません")
            return
        out_dir = os.path.join(self.take, "overlay")
        renderer = OverlayRenderer(seq, intr)
        n_drawn = renderer.snapshots(self.rgb, out_dir, 6)
        renderer.render_video(self.rgb, os.path.join(out_dir, "overlay.mp4"))
        print(f"[INFO] 静止画 {n_drawn} 枚出力。手の輪郭と点群の一致を目視確認してください（関門③）。")

    def _make_pseudo_intrinsics(self, out_path):
        """intrinsics 不在時: SLAM npz の推定焦点距離・画像中心から擬似 intrinsics を作る。"""
        import cv2
        npzs = sorted(glob.glob(os.path.join(self.seq_folder, "SLAM", "hawor_slam_w_scale_*.npz")))
        slam = dict(np.load(npzs[-1], allow_pickle=True))
        img = cv2.imread(sorted(glob.glob(
            os.path.join(self.seq_folder, "extracted_images", "*.jpg")))[0])
        H, W = img.shape[:2]
        intr = Intrinsics(
            width=W, height=H,
            fx=float(slam["img_focal"]), fy=float(slam["img_focal"]),
            cx=float(slam["img_center"][0]), cy=float(slam["img_center"][1]),
            model="pseudo_from_slam", coeffs=[], fps=None,
            extra={"note": "intrinsics 未指定のため SLAM 推定値から生成。投影確認専用・精度注意。"},
        )
        intr.save(out_path)
        print(f"[WARN] intrinsics.json が無いため擬似値を生成: fx={intr.fx:.0f}"
              f"（calibrate_camera.py の実測を推奨）")
        return out_path

    # ---- エクスポート（α=1.0 既定）----
    def export(self):
        cfg = self.cfg
        if not (cfg.force or not os.path.exists(self.traj_npz)):
            return
        fps = None
        if self.has_intr:
            fps = Intrinsics.load(self.intr).fps
        fps = fps or video_fps(self.rgb)
        seq = HaworSequence(self.seq_folder)
        TrajectoryExporter(seq).export(self.traj_npz, fps=fps)

    # ---- 3D 可視化（カメラ軌跡 + 手スケルトン）----
    def vis3d(self):
        cfg = self.cfg
        vis3d_mp4 = os.path.join(self.take, "vis3d", "world_3d.mp4")
        if cfg.skip_vis3d or not (cfg.force or not os.path.exists(vis3d_mp4)):
            return
        fps = video_fps(self.rgb) or 30.0
        d = np.load(self.traj_npz)
        joints, valid = d["joints"].astype(np.float64), d["valid_per_hand"]
        alpha = float(d["alpha"])
        seq = HaworSequence(self.seq_folder)
        cam_pos = seq.cam_centers * alpha   # 関節と同じ α でスケール
        renderer = Skeleton3DRenderer(joints, valid, cam_pos, seq.R_c2w, flip=True)
        n = renderer.render_video(vis3d_mp4, fps=fps, stride=1, size=(1280, 720),
                                  elev=20.0, azim=-60.0)
        out_fps = max(fps / 1, 1.0)
        print(f"[INFO] 3D 可視化動画: {vis3d_mp4} ({n} frames, {out_fps:.2f} fps, alpha={alpha})")

    # ---- 連結実行 ----
    def run(self):
        """全ステージを順に実行する。各段のスキップ判定は旧 run_pipeline.py と同一。"""
        self.ingest()
        self.rectify()
        self.infer()
        self.overlay()
        self.export()
        self.vis3d()

        print(f"\n[DONE] 最終出力: {self.traj_npz}")
        print(f"[DONE] 関門③'確認用: {os.path.join(self.take, 'overlay')}/*.png + overlay.mp4")
        if not self.cfg.skip_vis3d:
            print(f"[DONE] 3D 可視化: {os.path.join(self.take, 'vis3d', 'world_3d.mp4')}")
        return 0
