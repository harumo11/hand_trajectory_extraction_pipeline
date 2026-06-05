#!/usr/bin/env python3
"""visualization.py — world 軌道・MANO 投影の可視化（オーバーレイ / 3D スケルトン）。

- OverlayRenderer  : MANO 頂点をカメラ座標→画素へ投影し RGB に重畳（旧 tools/overlay_check.py）。
                     カメラ規約（+Z 前方・y 下・u=fx·X/Z+cx）の確認（関門③）に使う。
- Skeleton3DRenderer: world 座標の両手スケルトン＋カメラ姿勢を matplotlib で 3D 描画
                     （旧 tools/visualize_3d.py）。入力は npz 由来の α 適用済み素の配列。

カメラ投影は handtraj.camera.project、mp4 書き出しは handtraj.video_io.Mp4Writer、
ボーン接続は handtraj.hawor_adapter.HAND_BONES を利用する（自前定義しない）。
"""
import os

import cv2
import numpy as np

from handtraj.camera import project
from handtraj.hawor_adapter import HAND_BONES
from handtraj.video_io import Mp4Writer

# HaWoR demo と同じ表示用変換（y-up へ）
R_X = np.diag([1.0, -1.0, -1.0])

# カメラフラスタム（カメラ座標, m）: 原点(光学中心) + 像面四隅
FRUSTUM_C = np.array([[0, 0, 0],
                      [-0.04, -0.025, 0.06], [0.04, -0.025, 0.06],
                      [0.04, 0.025, 0.06], [-0.04, 0.025, 0.06]])
FRUSTUM_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]


def equal_limits(points, pad=0.10):
    """全点を包含する等方の軸範囲 (3,2) を返す。"""
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2
    half = (hi - lo).max() / 2 * (1 + pad) + 1e-6
    return np.stack([center - half, center + half], axis=1)


class OverlayRenderer:
    """MANO 頂点投影の RGB 重畳（旧 tools/overlay_check.py のロジック）。"""

    HAND_COLOR = {"left": (0, 0, 255), "right": (0, 255, 0)}  # BGR: 左=赤, 右=緑

    def __init__(self, seq, intr):
        self.seq = seq
        self.intr = intr
        self.K = np.asarray(intr.K)

    def draw_frame(self, img, t, label=None):
        """フレーム t の有効な手の頂点投影を img に描き込む（in-place）。"""
        H, W = img.shape[:2]
        verts_cam = self.seq.verts_cam
        valid = self.seq.valid
        for h, hand in enumerate(("left", "right")):
            if not valid[t, h]:
                continue
            uv, ok = project(verts_cam[t, h], self.K)
            u, v = uv[:, 0], uv[:, 1]
            ui = np.round(u[ok]).astype(int)
            vi = np.round(v[ok]).astype(int)
            inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            for x, y in zip(ui[inside], vi[inside]):
                cv2.circle(img, (x, y), 1, self.HAND_COLOR[hand], -1)
        cv2.putText(img, label if label is not None else f"frame {t}  (L=red R=green)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img

    def snapshots(self, video_path, out_dir, num=6):
        """等間隔サンプルの静止画を out_dir に書き出す。描いた枚数を返す。"""
        valid = self.seq.valid
        valid_idx = np.where(valid.any(axis=1))[0]
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        pick = valid_idx[np.linspace(0, len(valid_idx) - 1, min(num, len(valid_idx))).astype(int)]
        n_drawn = 0
        for t in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
            ret, img = cap.read()
            if not ret:
                print(f"[WARN] frame {t} 読込失敗")
                continue
            self.draw_frame(img, int(t))
            out_path = os.path.join(out_dir, f"overlay_{t:06d}.png")
            cv2.imwrite(out_path, img)
            n_drawn += 1
            print(f"[INFO] {out_path}")
        cap.release()
        return n_drawn

    def render_video(self, video_path, out_path):
        """全フレームを順次読みながら重畳して mp4 に書き出す。書き出したフレーム数を返す。"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = float(fps)
        writer = Mp4Writer(out_path, fps, (W, H))
        valid = self.seq.valid
        T = valid.shape[0]
        t = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break
            if t < T:
                self.draw_frame(img, t, label=f"frame {t}  (L=red R=green)")
            writer.write(img)
            t += 1
        writer.close()
        cap.release()
        print(f"[INFO] 重畳動画: {out_path} ({t} frames, {fps:.2f} fps)")
        return t


class Skeleton3DRenderer:
    """world 座標の 3D 表示（旧 tools/visualize_3d.py のロジック）。

    入力は HaworSequence ではなく素の配列（npz 由来の α 適用済み関節を受けるため）。
    flip=True で HaWoR demo と同じ R_x=diag(1,-1,-1) を掛けた y-up 系へ変換する。
    """

    HAND_COLOR = {"left": "red", "right": "green"}

    def __init__(self, joints, valid, cam_pos, R_c2w, flip=True):
        joints = np.asarray(joints, dtype=np.float64)
        valid = np.asarray(valid)
        cam_pos = np.asarray(cam_pos)
        R_c2w = np.asarray(R_c2w)

        T = min(joints.shape[0], cam_pos.shape[0])
        joints, valid = joints[:T], valid[:T]
        cam_pos, R_c2w = cam_pos[:T], R_c2w[:T]

        if flip:   # HaWoR demo と同じ y-up 表示
            joints = joints @ R_X.T
            cam_pos = cam_pos @ R_X.T
            R_c2w = np.einsum("ij,tjk->tik", R_X, R_c2w)

        self.joints = joints
        self.valid = valid
        self.cam_pos = cam_pos
        self.R_c2w = R_c2w
        self.T = T

        pts = [cam_pos]
        if valid.any():
            pts.append(joints[valid].reshape(-1, 3))
        self.lims = equal_limits(np.concatenate(pts, 0))

    def _draw_frame(self, ax, t, elev, azim, fps):
        joints, valid, cam_pos, R_c2w, lims = (
            self.joints, self.valid, self.cam_pos, self.R_c2w, self.lims)
        ax.cla()
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.view_init(elev=elev, azim=azim)
        time_s = f" ({t / fps:.2f}s)" if fps else ""
        ax.set_title(f"frame {t}{time_s}   camera=blue  L-hand=red  R-hand=green")

        # カメラ軌跡（全体は薄灰、現在までは青）
        ax.plot(*cam_pos.T, color="0.8", lw=0.8)
        ax.plot(*cam_pos[: t + 1].T, color="tab:blue", lw=1.5)

        # 現在のカメラ姿勢（フラスタム + 光軸）
        fr = (R_c2w[t] @ FRUSTUM_C.T).T + cam_pos[t]
        for i, j in FRUSTUM_EDGES:
            ax.plot(*np.stack([fr[i], fr[j]]).T, color="tab:blue", lw=1.2)
        axis_end = cam_pos[t] + R_c2w[t] @ np.array([0, 0, 0.09])
        ax.plot(*np.stack([cam_pos[t], axis_end]).T, color="tab:cyan", lw=1.0, ls=":")

        # 両手スケルトン + 手首軌跡
        for h, hand in enumerate(("left", "right")):
            c = self.HAND_COLOR[hand]
            wrist_path = joints[: t + 1, h, 0]
            ok = valid[: t + 1, h]
            if ok.any():
                wp = wrist_path.copy(); wp[~ok] = np.nan
                ax.plot(*wp.T, color=c, lw=0.7, alpha=0.35)
            if not valid[t, h]:
                continue
            J = joints[t, h]
            ax.scatter(*J.T, color=c, s=6, depthshade=False)
            for i, j in HAND_BONES:
                ax.plot(*np.stack([J[i], J[j]]).T, color=c, lw=1.5)

    def render_video(self, out_path, fps=30.0, stride=1,
                     size=(1280, 720), elev=20.0, azim=-60.0):
        """3D 可視化動画を out_path へ書き出す。書き出したフレーム数を返す。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        W, H = int(size[0]), int(size[1])
        dpi = 100
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        out_fps = max(fps / stride, 1.0)
        writer = Mp4Writer(out_path, out_fps, (W, H))
        frames = range(0, self.T, stride)
        for k, t in enumerate(frames):
            self._draw_frame(ax, t, elev, azim, fps)
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.write(buf[:, :, ::-1].copy())   # RGB -> BGR
            if (k + 1) % 100 == 0:
                print(f"[INFO] {k + 1}/{len(frames)} frames rendered")
        writer.close()
        plt.close(fig)
        return len(frames)
