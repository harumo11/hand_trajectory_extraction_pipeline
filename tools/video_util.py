#!/usr/bin/env python3
"""video_util.py — 互換シム（実体は handtraj.video_io.Mp4Writer へ移設済み）。

旧来の `from video_util import Mp4Writer` / `from tools.video_util import Mp4Writer` を
壊さないための再エクスポート。新規コードは handtraj.video_io を直接 import すること。
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from handtraj.video_io import Mp4Writer  # noqa: E402,F401
