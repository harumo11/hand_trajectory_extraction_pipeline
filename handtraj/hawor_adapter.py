#!/usr/bin/env python3
"""
hawor_adapter.py — HaWoR 出力（seq_folder）への高水準アクセス（CLAUDE.md §5）。

HaWoR の出力（seq_folder = <video_dir>/<video_name>/）:
  - world_space_res.pth                      … joblib: [pred_trans, pred_rot, pred_hand_pose,
                                                pred_betas, pred_valid]  各 [2, T, ...]（0=left, 1=right）
                                                world 座標・HaWoR尺度・メートル
  - SLAM/hawor_slam_w_scale_<s>_<e>.npz      … traj(c2w)+scale 等。load_slam_cam で R/t に変換

フレーム表現（§5 の表）:
  idx:int            depth_NNNNNN.png と一致（動画フレーム番号。ffmpeg -start_number 0 抽出と
                     split_bag.py の連番書き出しが共に 0 始まりで 1:1 対応）
  verts_cam_m:(778,3) MANO頂点（カメラ座標・m・HaWoR尺度）
  faces:(F,3)        MANO面（手首閉鎖の追加面を含む。demo.py と同一構成）
  world_joints:(2,21,3) world手関節（HaWoR尺度、[left, right]）
  hand               'left' / 'right'

カメラ規約: DROID-SLAM / HaWoR とも OpenCV系（+Z 前方・y 下・u=fx·X/Z+cx）を仮定。
           関門③（メッシュ投影オーバーレイ）で実データ検証すること（tools/overlay_check.py）。

環境変数:
  M3_FRAME_STRIDE … M3 較正用フレームの間引き幅。未指定時は総フレームが
                    M3_MAX_FRAMES(既定300) 以下になるよう自動決定（α はグローバル median のため
                    部分フレームで統計的に十分。純Pythonレンダの速度対策）。

重い計算（MANO forward / 座標変換）は初回アクセス時に1度だけ実行しキャッシュする。
"""
import glob
import os
import sys
from contextlib import contextmanager

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HAWOR_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "third_party", "HaWoR"))

HAND_ORDER = ("left", "right")   # demo.py の hand2idx = {right:1, left:0} に一致

# MANO 関節順（OpenPose 互換, アダプタ出力）: 0=wrist, 9=middle_MCP, 12=middle_tip
WRIST, MIDDLE_MCP, MIDDLE_TIP = 0, 9, 12

# OpenPose 手関節順のボーン接続（0=wrist, 1-4=親指, 5-8=人差指, 9-12=中指, 13-16=薬指, 17-20=小指）
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
              (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20)]

# demo.py / run_mano と同一の手首閉鎖追加面（頂点は 778 のまま増えない）
FACES_NEW = np.array([
    [92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
    [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
    [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
    [120, 108, 78], [78, 108, 79]])


@contextmanager
def _hawor_cwd():
    """HaWoR 内部は '_DATA/...' 等の cwd 相対パス前提のため、HaWoR ルートで実行する。"""
    prev = os.getcwd()
    os.chdir(HAWOR_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def _import_hawor():
    if HAWOR_ROOT not in sys.path:
        sys.path.insert(0, HAWOR_ROOT)


def _find_slam_npz(seq_folder):
    files = sorted(glob.glob(os.path.join(seq_folder, "SLAM", "hawor_slam_w_scale_*.npz")))
    if not files:
        raise FileNotFoundError(f"SLAM 出力が見つかりません: {seq_folder}/SLAM/hawor_slam_w_scale_*.npz")
    return files[-1]


def _run_mano_chunked(fn, trans, rot, pose, betas, use_cuda, chunk=1024):
    """run_mano(_left) を時間方向チャンクで実行（長尺でのGPUメモリ対策）。numpy [T,...] を返す。"""
    import torch
    T = trans.shape[1]
    joints, verts = [], []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        out = fn(trans[:, s:e], rot[:, s:e], pose[:, s:e], betas=betas[:, s:e], use_cuda=use_cuda)
        joints.append(out["joints"][0].detach().cpu().numpy())
        verts.append(out["vertices"][0].detach().cpu().numpy())
        if use_cuda:
            torch.cuda.empty_cache()
    return np.concatenate(joints, 0), np.concatenate(verts, 0)


def _decide_stride(n_valid_frames):
    stride = os.environ.get("M3_FRAME_STRIDE")
    if stride:
        return max(1, int(stride))
    max_frames = int(os.environ.get("M3_MAX_FRAMES", "300"))
    return max(1, int(np.ceil(n_valid_frames / max_frames)))


class HaworSequence:
    """HaWoR の seq_folder 出力（world_space_res.pth + SLAM npz）への高水準アクセス。

    重い計算（MANO forward / 座標変換）は初回アクセス時に1度だけ実行しキャッシュ。
    各プロパティは numpy で返す（遅延評価）。
    """

    def __init__(self, seq_folder):
        self.seq_folder = os.path.abspath(seq_folder)
        self._outputs = None   # load_hawor_outputs 相当のキャッシュ

    # ---- 内部: 重い読み込み（初回のみ）。旧 load_hawor_outputs のロジックを移設 ----
    def _load(self):
        """HaWoR 出力一式を numpy で読み込みキャッシュする（world 頂点/関節・カメラ外部パラメータ・valid）。

        keys:
          joints_world [T,2,21,3], verts_world [T,2,778,3]  … m, HaWoR尺度, world座標
          verts_cam    [T,2,778,3]                          … m, HaWoR尺度, カメラ座標(OpenCV系)
          faces        {'left':(F,3),'right':(F,3)}
          valid        [T,2] bool（0=left,1=right）
          R_w2c [T,3,3], t_w2c [T,3]（world→camera）
        """
        if self._outputs is not None:
            return self._outputs

        import joblib
        import torch
        _import_hawor()
        from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
        from lib.eval_utils.custom_utils import load_slam_cam

        seq_folder = self.seq_folder
        res_path = os.path.join(seq_folder, "world_space_res.pth")
        if not os.path.exists(res_path):
            raise FileNotFoundError(f"HaWoR 結果がありません: {res_path}（先に HaWoR 推論を実行）")
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(res_path)

        R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(_find_slam_npz(seq_folder))

        use_cuda = torch.cuda.is_available()
        with _hawor_cwd():
            # 0=left は専用の run_mano_left（shapedirs 修正込み）、1=right は run_mano
            jl, vl = _run_mano_chunked(run_mano_left, pred_trans[0:1], pred_rot[0:1],
                                       pred_hand_pose[0:1], pred_betas[0:1], use_cuda)
            jr, vr = _run_mano_chunked(run_mano, pred_trans[1:2], pred_rot[1:2],
                                       pred_hand_pose[1:2], pred_betas[1:2], use_cuda)
            faces_right = np.concatenate([np.asarray(get_mano_faces(), dtype=np.int64), FACES_NEW], 0)
        faces_left = faces_right[:, [0, 2, 1]]

        joints_world = np.stack([jl, jr], axis=1)   # [T,2,21,3]
        verts_world = np.stack([vl, vr], axis=1)    # [T,2,778,3]

        Rw2c = R_w2c.numpy().astype(np.float64)     # [T,3,3]
        tw2c = t_w2c.numpy().astype(np.float64)     # [T,3]
        T = min(len(Rw2c), verts_world.shape[0])
        joints_world, verts_world = joints_world[:T], verts_world[:T]
        Rw2c, tw2c = Rw2c[:T], tw2c[:T]

        # world → camera（OpenCV系を仮定。関門③で検証）
        verts_cam = np.einsum("tij,thnj->thni", Rw2c, verts_world) + tw2c[:, None, None, :]

        valid = (np.asarray(pred_valid)[:, :T].T > 0)   # [T,2]
        self._outputs = {
            "joints_world": joints_world,
            "verts_world": verts_world,
            "verts_cam": verts_cam,
            "faces": {"left": faces_left, "right": faces_right},
            "valid": valid,
            "R_w2c": Rw2c,
            "t_w2c": tw2c,
        }
        return self._outputs

    # ---- プロパティ（すべて numpy・遅延評価）----
    @property
    def joints_world(self):
        """world 手関節 [T,2,21,3]（m, HaWoR尺度）。"""
        return self._load()["joints_world"]

    @property
    def verts_world(self):
        """world MANO頂点 [T,2,778,3]（m, HaWoR尺度）。"""
        return self._load()["verts_world"]

    @property
    def verts_cam(self):
        """カメラ座標 MANO頂点 [T,2,778,3]（m, HaWoR尺度, OpenCV系）。"""
        return self._load()["verts_cam"]

    @property
    def valid(self):
        """valid マスク [T,2] bool（0=left, 1=right）。"""
        return self._load()["valid"]

    @property
    def faces(self):
        """MANO面 {'left':(F,3) int64, 'right':(F,3)}。"""
        return self._load()["faces"]

    @property
    def R_w2c(self):
        """world→camera 回転 [T,3,3]。"""
        return self._load()["R_w2c"]

    @property
    def t_w2c(self):
        """world→camera 並進 [T,3]。"""
        return self._load()["t_w2c"]

    @property
    def R_c2w(self):
        """camera→world 回転 [T,3,3]（R_w2c の転置で導出）。"""
        return np.transpose(self._load()["R_w2c"], (0, 2, 1))

    @property
    def cam_centers(self):
        """world 系のカメラ光学中心 [T,3]（-R_c2w @ t_w2c）。"""
        out = self._load()
        R_c2w = np.transpose(out["R_w2c"], (0, 2, 1))
        return -np.einsum("tij,tj->ti", R_c2w, out["t_w2c"])

    @property
    def T(self):
        """フレーム数 int。"""
        return int(self._load()["joints_world"].shape[0])

    def frames(self, stride=None, max_frames=None):
        """§5 形式の dict リストを返す（旧 load_hawor_frames と同一形式）。

        1フレームにつき valid な手ごとに 1 エントリ（左右 valid なら 2 エントリ）。
        純Pythonレンダの速度対策で間引く（α推定には十分）。
        間引き既定は環境変数 M3_FRAME_STRIDE / M3_MAX_FRAMES(既定300) を尊重するが、
        引数 stride / max_frames が与えられた場合はそれらで上書きする。
        """
        data = self._load()
        valid = data["valid"]
        valid_idx = np.where(valid.any(axis=1))[0]
        if stride is not None:
            use_stride = max(1, int(stride))
        elif max_frames is not None:
            use_stride = max(1, int(np.ceil(len(valid_idx) / int(max_frames))))
        else:
            use_stride = _decide_stride(len(valid_idx))
        use_idx = valid_idx[::use_stride]
        print(f"[adapter] frames: total={valid.shape[0]} valid={len(valid_idx)} "
              f"stride={use_stride} -> M3 使用 {len(use_idx)} フレーム")

        frames = []
        for t in use_idx:
            for h, hand in enumerate(HAND_ORDER):
                if not valid[t, h]:
                    continue
                frames.append({
                    "idx": int(t),
                    "verts_cam_m": data["verts_cam"][t, h].astype(np.float64),
                    "faces": data["faces"][hand],
                    "world_joints": data["joints_world"][t].astype(np.float64),
                    "hand": hand,
                })
        return frames

    def world_joints(self):
        """M3+ 用: world 手関節（HaWoR尺度）と valid を返す（旧 load_world_joints と同一）。

        Returns:
          joints [T,2,21,3] (m, HaWoR尺度。invalid な手は NaN)
          valid  [T,2] bool
        """
        data = self._load()
        joints = data["joints_world"].copy()
        valid = data["valid"]
        joints[~valid] = np.nan
        return joints, valid
