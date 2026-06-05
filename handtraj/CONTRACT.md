# handtraj パッケージ API 契約書（リファクタリング作業用・厳守）

並行作業する全エージェントはこの契約に**正確に**従うこと。シグネチャの独自変更禁止。
コメント・docstring は既存コードと同様に日本語。numpy<2 / torch 2.7.1 環境。
プロジェクトルート: /home/daiv02/repos/python/human_hand_video_to_vla_dataset

## handtraj/camera.py（担当: core）

```python
from dataclasses import dataclass, field

@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = ""
    coeffs: list = field(default_factory=list)
    fps: float | None = None
    extra: dict = field(default_factory=dict)   # calib_rms_px / note 等の残りキーを保持

    @classmethod
    def load(cls, path: str) -> "Intrinsics": ...      # JSON から（未知キーは extra へ）
    def save(self, path: str) -> None: ...             # JSON へ（extra のキーも書き戻す）
    @property
    def K(self) -> "np.ndarray": ...                   # 3x3 float64
    def scaled_to(self, width: int, height: int) -> "Intrinsics": ...
        # 解像度比例スケール（fx*sx, fy*sy, cx*sx, cy*sy）。アスペクト比が1%超ずれたら警告print
    def needs_rectify(self, tol_px: float = 1.0, tol_f: float = 1e-3) -> bool: ...
    def rectified(self) -> "Intrinsics": ...           # f=(fx+fy)/2, c=画像中央, coeffs=0, model="rectified_pinhole"

def project(pts_cam: "np.ndarray", K: "np.ndarray") -> tuple:
    """pts_cam[...,3] (OpenCV系カメラ座標, m) を K で投影。
    Returns: (uv[...,2], valid[...])  valid は Z>1e-6"""

class VideoRectifier:
    def __init__(self, intr: Intrinsics): ...
    def rectify_video(self, video_in: str, video_out: str) -> tuple:
        """理想ピンホール化して書き出し。Returns: (n_frames:int, intr_rect:Intrinsics)"""

class CameraCalibrator:
    """チェスボードキャリブレーション（旧 calibrate_camera.py のロジックを移設）"""
    AUTO_PATTERNS = [(5,4),(8,5),(9,6),(7,6),(6,5),(4,5),(5,8),(6,9),(6,7),(5,6)]
    def __init__(self, pattern: tuple | None = None, square_mm: float = 27.0): ...
    def detect_pattern(self, images: list) -> tuple: ...
    def calibrate_images(self, images: list, fps=None) -> Intrinsics: ...
        # 内部で collect_corners 相当 + cv2.calibrateCamera。
        # rms は Intrinsics.extra["calib_rms_px"], views は extra["calib_views"] に格納
    def calibrate_video(self, video_path: str, max_frames: int = 30) -> Intrinsics: ...
```

## handtraj/video_io.py（担当: core）

```python
class Mp4Writer:   # 現 tools/video_util.py の実装をそのまま移設（挙動変更禁止）
    def __init__(self, out_path: str, fps: float, size_wh: tuple): ...
    def write(self, bgr_frame) -> None: ...
    def close(self) -> None: ...

def video_fps(path: str) -> float | None: ...
def video_size(path: str) -> tuple: ...        # (W, H)
def read_frame(path: str, idx: int): ...       # BGR ndarray | None
def iter_frames(path: str): ...                # 順次 (idx, frame) を yield
```

## handtraj/hawor_adapter.py（担当: adapter）

```python
HAWOR_ROOT: str    # third_party/HaWoR の絶対パス
HAND_ORDER = ("left", "right")
# OpenPose 手関節順の定義
WRIST, MIDDLE_MCP, MIDDLE_TIP = 0, 9, 12
HAND_BONES = [...20本...]   # 現 tools/visualize_3d.py の定義を移設
FACES_NEW = ...             # 現 adapters/load_hawor_frames.py の定義を移設

class HaworSequence:
    """HaWoR の seq_folder 出力（world_space_res.pth + SLAM npz）への高水準アクセス。
    重い計算（MANO forward / 座標変換）は初回アクセス時に1度だけ実行しキャッシュ。"""
    def __init__(self, seq_folder: str): ...
    # 以下プロパティ（すべて numpy・遅延評価）
    joints_world  # [T,2,21,3] m HaWoR尺度
    verts_world   # [T,2,778,3]
    verts_cam     # [T,2,778,3] OpenCV系
    valid         # [T,2] bool
    faces         # {"left": (F,3) int64, "right": (F,3)}
    R_w2c, t_w2c  # [T,3,3], [T,3]
    R_c2w         # [T,3,3]（転置で導出）
    cam_centers   # [T,3]  -R_c2w @ t_w2c
    T             # int フレーム数
    def frames(self, stride: int | None = None, max_frames: int | None = None) -> list:
        """§5 形式の dict リスト（旧 load_hawor_frames と同一形式・同一の自動間引き既定:
        環境変数 M3_FRAME_STRIDE / M3_MAX_FRAMES(既定300) を尊重）"""
    def world_joints(self) -> tuple:
        """(joints[T,2,21,3] invalid→NaN, valid[T,2])  旧 load_world_joints と同一"""

# 内部実装は現 adapters/load_hawor_frames.py の _hawor_cwd / _import_hawor /
# _run_mano_chunked / load_hawor_outputs のロジックを移設（数値挙動を変えないこと）
```

## handtraj/export.py（担当: adapter）

```python
class TrajectoryExporter:
    def __init__(self, seq: "HaworSequence"): ...
    def export(self, out_npz: str, alpha: float = 1.0, m3_report: str | None = None,
               fps: float | None = None) -> dict:
        """world_joints × α → npz 保存（keys: joints/alpha/valid/valid_per_hand 現行と同一）。
        m3_report 指定時は alpha_global を優先し alpha_per_frame.png を出力。
        Returns: {"alpha":..., "T":..., "n_valid":..., "hand_length_m": {"left":..,"right":..}}
        手長チェックの [CHECK] print も現行 export_trajectory.py と同様に行う。"""

def hand_length_stats(joints, valid) -> dict: ...   # 現実装を移設
```

## handtraj/visualization.py（担当: vis）

```python
class OverlayRenderer:
    """MANO 頂点投影の RGB 重畳（旧 tools/overlay_check.py のロジック）"""
    HAND_COLOR = {"left": (0,0,255), "right": (0,255,0)}
    def __init__(self, seq: "HaworSequence", intr: "Intrinsics"): ...
    def draw_frame(self, img, t: int, label: str | None = None): ...   # in-place
    def snapshots(self, video_path: str, out_dir: str, num: int = 6) -> int: ...
    def render_video(self, video_path: str, out_path: str) -> int: ...

class Skeleton3DRenderer:
    """world 座標の 3D 表示（旧 tools/visualize_3d.py のロジック）。
    入力は HaworSequence ではなく素の配列（npz 由来の α 適用済み関節を受けるため）"""
    def __init__(self, joints, valid, cam_pos, R_c2w, flip: bool = True): ...
    def render_video(self, out_path: str, fps: float = 30.0, stride: int = 1,
                     size=(1280,720), elev: float = 20.0, azim: float = -60.0) -> int: ...
# 依存: handtraj.camera.project / handtraj.video_io.Mp4Writer / handtraj.hawor_adapter.HAND_BONES
```

## handtraj/pipeline.py（担当: pipeline・第2波）

```python
class PipelineConfig:  # dataclass: take, video=None, intrinsics=None, force=False,
                       # skip_overlay=False, skip_vis3d=False, no_rectify=False
class Pipeline:
    def __init__(self, cfg: PipelineConfig): ...
    def ingest(self): ...      # 動画/intrinsics 配置 + 解像度調整
    def rectify(self): ...     # needs_rectify なら VideoRectifier（.rectified マーカー管理含む）
    def infer(self): ...       # tools/hawor_infer.py を subprocess で（現行と同じ・プロセス分離維持）
    def overlay(self): ...     # OverlayRenderer（intrinsics 無ければ SLAM 由来の擬似値生成）
    def export(self): ...      # TrajectoryExporter
    def vis3d(self): ...       # Skeleton3DRenderer
    def run(self): ...         # 上記を順に。各段スキップ判定は現 run_pipeline.py と同一挙動
```

## CLI（シンウラッパ・引数と出力は現行完全互換）
- calibrate_camera.py（core 担当）: argparse + CameraCalibrator。--selftest は現行ロジック維持（2ケース）
- record_rgb.py（core 担当）: 現行ロジックを軽整理（クラス化は任意・挙動不変）
- tools/rectify_video.py（core 担当）: VideoRectifier 利用のシンラッパ（needs_rectify 表示含む）
- export_trajectory.py（adapter 担当）: TrajectoryExporter 利用のシンラッパ
- adapters/load_hawor_frames.py（adapter 担当）: 互換シム。load_hawor_frames / load_world_joints /
  load_hawor_outputs / HAWOR_ROOT / FACES_NEW / HAND_ORDER を handtraj から再エクスポート
  （m3_surface_calibrate.py 等の既存 import を壊さない。docstring に移設先を明記）
- tools/overlay_check.py, tools/visualize_3d.py（vis 担当）: クラス利用のシンラッパ（引数互換）
- tools/video_util.py（vis 担当）: `from handtraj.video_io import Mp4Writer` の互換シム
- run_pipeline.py（pipeline 担当）: argparse → PipelineConfig → Pipeline.run()

## handtraj/refine.py（2026-06 追加: 2D キーポイント並進リファイン）

```python
class KeypointObservations:
    def __init__(self, npz_path): ...        # tools/detect_keypoints_2d.py の出力を読む
    def match_and_gate(self, uv_init, valid, gate_px=80.0) -> "np.ndarray": ...
        # 左右スワップ補正 + 残差中央値ゲート。use[T,2] を返す

class RefineResult:
    delta_t_cam   # [T,2,3] m（カメラ座標系の並進補正）
    delta_t_world # [T,2,3] m（world 座標系）
    use           # [T,2] 最適化に使用した観測
    diagnostics   # dict（残差 before/after・swap/gate 件数・|Δt| 統計）
    def apply_world(self, joints_world): ...

class TranslationRefiner:
    def __init__(self, joints_cam, valid, K, R_c2w): ...
    def solve(self, obs, gate_px=80.0, sigma_prior_m=0.10, sigma_vel_m=0.003,
              huber_px=10.0, iters=1500, lr=5e-4, fit_mask=None, verbose=True) -> RefineResult: ...
        # fit_mask[T]: ホールドアウト評価用（True のフレームのみ当てはめ）

def refine_take(take_dir, keypoints=None, sigma_vel_m=0.003, gate_px=80.0,
                skip_overlay=False) -> dict: ...
    # テイク一式の高水準API（CLI/Pipeline 共用）。world_trajectory.npz を更新
    # （追加キー: refined bool / delta_t_world [T,2,3] float32。既存キー形式は不変）
def refine_npz(base_npz, result, out_npz=None, report_json=None) -> str: ...
def run_selftest() -> int: ...               # 関門④a（面内復元≥90% 等）

# 検出側（隔離 venv .venv_mp で実行・handtraj 非依存）: tools/detect_keypoints_2d.py
#   出力 npz: k2d[T,2,21,2](NaN=未検出) / detected[T,2] / score[T,2] / width/height/n_frames
# PipelineConfig に refine: bool = False を追加（検出→refine_take を export 後に実行）
```

## 共通ルール
1. sys.path: ルート直下スクリプトは `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`、
   tools/ 配下は親ディレクトリを挿入してから `import handtraj.*`
2. 触ってはいけない: third_party/ 全体, record_d405.py, split_bag.py, m3_surface_calibrate.py,
   tools/hawor_infer.py, tools/prepare_mano.py, captures/, weights, *.json, *.npz
3. print メッセージ（[INFO]/[WARN]/[CHECK] 等）は現行文言を維持（ログ互換）
4. 自己検証: 担当ファイル全部に `.venv/bin/python -m py_compile` を通すこと。
   handtraj パッケージ全体の import は第2波（pipeline 担当）が確認するので、
   第1波は自モジュール単体の compile + 可能な範囲の単体 import まででよい
5. handtraj/__init__.py は pipeline 担当が最後に作成（主要クラスの再エクスポート）
