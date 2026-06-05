#!/usr/bin/env python3
"""handtraj — 両手・指 world 軌道推定システムの中核パッケージ（CLAUDE.md）。

主要な公開クラス・関数を再エクスポートする。

import 時に torch を読み込まない設計を維持するため、各サブモジュールは
torch を遅延 import する（hawor_adapter は MANO forward 等を初回アクセス時のみ実行）。
このパッケージの import で重い依存（torch/CUDA）が走らないことが前提。
"""
from handtraj.camera import (
    CameraCalibrator,
    Intrinsics,
    VideoRectifier,
    project,
)
from handtraj.export import TrajectoryExporter
from handtraj.hawor_adapter import HaworSequence
from handtraj.pipeline import Pipeline, PipelineConfig
from handtraj.refine import KeypointObservations, TranslationRefiner, refine_take
from handtraj.video_io import Mp4Writer
from handtraj.visualization import OverlayRenderer, Skeleton3DRenderer

__all__ = [
    "Intrinsics",
    "project",
    "VideoRectifier",
    "CameraCalibrator",
    "Mp4Writer",
    "HaworSequence",
    "TrajectoryExporter",
    "OverlayRenderer",
    "Skeleton3DRenderer",
    "Pipeline",
    "PipelineConfig",
    "KeypointObservations",
    "TranslationRefiner",
    "refine_take",
]
