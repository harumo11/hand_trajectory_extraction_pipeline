#!/usr/bin/env python3
"""
load_hawor_frames.py — HaWoR アダプタの互換シム（CLAUDE.md §5）。

★本体ロジックは handtraj.hawor_adapter.HaworSequence へ移設済み★
本ファイルは既存スクリプト（m3_surface_calibrate.py / tools/visualize_3d.py 等）の import を
壊さないための薄い互換シム。公開名 load_hawor_frames / load_world_joints / load_hawor_outputs /
HAWOR_ROOT / FACES_NEW / HAND_ORDER は従来と同じ入出力（dict のキー・shape・dtype を含め完全一致）。

新規コードは handtraj.hawor_adapter.HaworSequence を直接使うこと。

フレーム表現（§5 の表）:
  idx:int            depth_NNNNNN.png と一致（動画フレーム番号）
  verts_cam_m:(778,3) MANO頂点（カメラ座標・m・HaWoR尺度）
  faces:(F,3)        MANO面（手首閉鎖の追加面を含む。demo.py と同一構成）
  world_joints:(2,21,3) world手関節（HaWoR尺度、[left, right]）
  hand               'left' / 'right'

カメラ規約: DROID-SLAM / HaWoR とも OpenCV系（+Z 前方・y 下・u=fx·X/Z+cx）を仮定。

環境変数:
  M3_FRAME_STRIDE / M3_MAX_FRAMES … M3 較正用フレームの間引き（既定動作は移設先と同一）。
"""
import os
import sys

# 親ディレクトリ（プロジェクトルート）を path に入れて handtraj を import 可能にする
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from handtraj.hawor_adapter import (  # noqa: E402
    HAND_ORDER,
    HAWOR_ROOT,
    FACES_NEW,
    HaworSequence,
)

__all__ = [
    "HAND_ORDER",
    "HAWOR_ROOT",
    "FACES_NEW",
    "load_hawor_outputs",
    "load_hawor_frames",
    "load_world_joints",
]


def load_hawor_outputs(seq_folder):
    """HaWoR 出力一式を numpy で返す（旧 API 互換）。

    実体は HaworSequence._load()。返り値 dict のキー・shape・dtype は従来と完全一致:
      joints_world [T,2,21,3], verts_world [T,2,778,3], verts_cam [T,2,778,3],
      faces {'left':(F,3),'right':(F,3)}, valid [T,2] bool, R_w2c [T,3,3], t_w2c [T,3]
    """
    return HaworSequence(seq_folder)._load()


def load_hawor_frames(hawor_output_path):
    """§5 仕様のフレーム表現リストを返す（旧 API 互換・m3_surface_calibrate.py 入力）。

    実体は HaworSequence.frames()（間引き既定: M3_FRAME_STRIDE / M3_MAX_FRAMES）。
    """
    return HaworSequence(hawor_output_path).frames()


def load_world_joints(hawor_output_path):
    """M3+ 用: world 手関節（HaWoR尺度）と valid を返す（旧 API 互換）。

    実体は HaworSequence.world_joints()。
    Returns:
      joints [T,2,21,3] (m, HaWoR尺度。invalid な手は NaN)
      valid  [T,2] bool
    """
    return HaworSequence(hawor_output_path).world_joints()
