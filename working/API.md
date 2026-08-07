# handtraj API リファレンス

`handtraj` パッケージの公開 API 仕様。**実コードから起こしたもの**（旧 `handtraj/CONTRACT.md` は
リファクタ時の作業指示書だったため、本ファイルで現行仕様に置き換えた）。

関連: [SOURCE_INDEX.md](SOURCE_INDEX.md)（ファイル索引）・[ARCHITECTURE.md](ARCHITECTURE.md)（実行フロー）

環境: numpy<2 / torch 2.7.1 / Python 3.12。すべての配列は numpy。

---

## 規約（恒久ルール）

1. **sys.path**: ルート直下のスクリプトは
   `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`、
   `tools/` 配下は親ディレクトリを挿入してから `import handtraj.*`
2. **`import handtraj` で torch/CUDA を走らせない。** サブモジュールは eager import されるが、
   torch は各モジュール内で遅延 import する設計。これを壊さないこと
3. **print 文言はログ互換のため維持**（`[INFO]` / `[WARN]` / `[CHECK]` / `[RUN]` / `[DONE]` / `[ERROR]`）
4. コメント・docstring は日本語（既存コードに合わせる）
5. `third_party/` は `patches/` 経由の最小パッチ以外改変しない（CLAUDE.md「方針（厳守）」）

## `handtraj/__init__.py` が再エクスポートする 14 名

`Intrinsics` `project` `VideoRectifier` `CameraCalibrator` `Mp4Writer` `HaworSequence`
`TrajectoryExporter` `OverlayRenderer` `Skeleton3DRenderer` `Pipeline` `PipelineConfig`
`KeypointObservations` `TranslationRefiner` `refine_take`

（`RefineResult` `refine_npz` `hand_length_stats` `equal_limits` `video_fps` `video_size`
`read_frame` `iter_frames` はサブモジュールから直接 import する）

---

## `handtraj/camera.py`

```python
@dataclass
class Intrinsics:
    width: int; height: int
    fx: float; fy: float; cx: float; cy: float
    model: str = ""
    coeffs: list = field(default_factory=list)
    fps: float | None = None
    extra: dict = field(default_factory=dict)   # calib_rms_px / calib_views / note 等

    @classmethod
    def load(cls, path) -> Intrinsics      # JSON から（未知キーは extra へ）
    def save(self, path) -> None           # JSON へ（extra のキーも書き戻す）
    @property
    def K(self) -> np.ndarray              # 3x3 float64
    def scaled_to(self, width, height) -> Intrinsics
        # 解像度比例スケール。アスペクト比が1%超ずれたら [WARN] を print し note を追記
    def needs_rectify(self, tol_px=1.0, tol_f=1e-3) -> bool
    def rectified(self) -> Intrinsics
        # f=(fx+fy)/2, c=画像中央, coeffs=0, model="rectified_pinhole"

def project(pts_cam, K) -> (uv, valid)
    """pts_cam[...,3]（OpenCV 系カメラ座標・m）を K で投影。valid は Z>1e-6"""

class VideoRectifier:
    def __init__(self, intr: Intrinsics)
    def rectify_video(self, video_in, video_out) -> (n_frames: int, intr_rect: Intrinsics)

class CameraCalibrator:
    AUTO_PATTERNS = [(5,4),(8,5),(9,6),(7,6),(6,5),(4,5),(5,8),(6,9),(6,7),(5,6)]
    def __init__(self, pattern=None, square_mm=27.0)
    def detect_pattern(self, images, n_probe=6) -> tuple
    def detect_pattern_or_self(self, images) -> tuple
    def calibrate_images(self, images, fps=None, show_progress=True) -> Intrinsics
        # rms → Intrinsics.extra["calib_rms_px"], views → extra["calib_views"]
    def calibrate_video(self, video_path, max_frames=30) -> Intrinsics
```

**カメラ規約は OpenCV 系**: +Z 前方・y 下・`u = fx·X/Z + cx`（v1 関門③で実証済み）。

---

## `handtraj/video_io.py`

```python
class Mp4Writer:
    def __init__(self, out_path, fps, size_wh)
    def write(self, bgr_frame) -> None
    def close(self) -> None          # close 時に ffmpeg で H.264 再エンコード

def video_fps(path) -> float | None
def video_size(path) -> (W, H)
def read_frame(path, idx)            # BGR ndarray | None
def iter_frames(path)                # (idx, frame) を順次 yield
```

---

## `handtraj/hawor_adapter.py`

```python
HAWOR_ROOT: str                      # third_party/HaWoR の絶対パス
HAND_ORDER = ("left", "right")       # demo.py の hand2idx = {right:1, left:0} に一致
WRIST, MIDDLE_MCP, MIDDLE_TIP = 0, 9, 12    # OpenPose 手関節順
HAND_BONES: list                     # 20 本のボーン接続
FACES_NEW: np.ndarray                # 手首閉鎖面（これを足して F=1552）

class HaworSequence:
    """HaWoR の seq_folder 出力（world_space_res.pth + SLAM npz）への高水準アクセス。
    重い計算（MANO forward / 座標変換）は初回アクセス時に1度だけ実行しキャッシュ。"""
    def __init__(self, seq_folder)

    # プロパティ（すべて numpy・遅延評価）
    joints_world   # [T,2,21,3] m・HaWoR 尺度
    verts_world    # [T,2,778,3]
    verts_cam      # [T,2,778,3]・OpenCV 系
    valid          # [T,2] bool
    faces          # {"left": (F,3) int64, "right": (F,3)}
    R_w2c, t_w2c   # [T,3,3], [T,3]
    R_c2w          # [T,3,3]（転置で導出）
    cam_centers    # [T,3]  = -R_c2w @ t_w2c
    T              # int フレーム数

    def frames(self, stride=None, max_frames=None) -> list[dict]
        """環境変数 M3_FRAME_STRIDE / M3_MAX_FRAMES（既定 300）を尊重して自動間引き"""
    def world_joints(self) -> (joints[T,2,21,3] invalid→NaN, valid[T,2])
```

### `frames()` が返す dict の形式（旧 §5 のフレーム表現・不変）

| フィールド | 用途 | 内容 |
|---|---|---|
| `idx: int` | 同期 | 動画フレーム番号（0 始まり） |
| `verts_cam_m: (778,3)` | 検証／将来の較正 | MANO 頂点（カメラ座標・m・HaWoR 尺度） |
| `faces: (F,3)` | 検証／将来の較正 | MANO 面（手首閉鎖面込み F=1552） |
| `world_joints: (2,21,3)` | エクスポート | world 手関節（HaWoR 尺度、`[left, right]`） |
| `hand` | 補助 | `left` / `right` |

---

## `handtraj/export.py`

```python
class TrajectoryExporter:
    def __init__(self, seq: HaworSequence)
    def export(self, out_npz, alpha=1.0, m3_report=None, fps=None) -> dict
        """world_joints × α → npz 保存。
        m3_report 指定時は alpha_global を優先し alpha_per_frame.png を出力。
        alpha は明示値なら「指定値で適用」、None なら既定 1.0（CLI の --alpha 省略時と一致）。
        Returns: {"alpha":.., "T":.., "n_valid":.., "hand_length_m": {"left":.., "right":..}}
        手長チェックの [CHECK] print も行う。"""

def hand_length_stats(joints, valid) -> dict
```

### `world_trajectory.npz` のキー

| キー | 型 | 内容 |
|---|---|---|
| `joints` | `[T,2,21,3]` float | world 手関節（m）。invalid は NaN |
| `alpha` | float | グローバルスケール係数（現行 1.0） |
| `valid` | `[T]` bool | どちらかの手が有効 |
| `valid_per_hand` | `[T,2]` bool | 手ごとの有効フラグ（`[left, right]`） |
| `refined` | bool | リファイン適用済みか（`refine_npz` が追加） |
| `delta_t_world` | `[T,2,3]` float32 | 適用した world 並進補正（`refine_npz` が追加） |

---

## `handtraj/visualization.py`

```python
def equal_limits(points, pad=0.10)

class OverlayRenderer:
    """MANO 頂点投影の RGB 重畳"""
    HAND_COLOR = {"left": (0,0,255), "right": (0,255,0)}
    def __init__(self, seq: HaworSequence, intr: Intrinsics)
    def draw_frame(self, img, t, label=None)              # in-place
    def snapshots(self, video_path, out_dir, num=6) -> int
    def render_video(self, video_path, out_path) -> int

class Skeleton3DRenderer:
    """world 座標の 3D 表示。入力は HaworSequence ではなく素の配列
    （npz 由来の α 適用済み関節を受けるため）"""
    def __init__(self, joints, valid, cam_pos, R_c2w, flip=True)
    def render_video(self, out_path, fps=30.0, stride=1,
                     size=(1280,720), elev=20.0, azim=-60.0) -> int
```

---

## `handtraj/refine.py`

```python
class KeypointObservations:
    def __init__(self, npz_path)                 # tools/detect_keypoints_2d.py の出力を読む
    def match_and_gate(self, uv_init, valid, gate_px=80.0) -> np.ndarray
        """左右スワップ補正 + 残差中央値ゲート。use[T,2] を返す"""

class RefineResult:
    delta_t_cam    # [T,2,3] m（カメラ座標系の並進補正）
    delta_t_world  # [T,2,3] m（world 座標系）
    use            # [T,2] 最適化に使用した観測
    diagnostics    # dict（残差 before/after・swap/gate 件数・|Δt| 統計）
    def apply_world(self, joints_world)

class TranslationRefiner:
    def __init__(self, joints_cam, valid, K, R_c2w)
    def project(self, pts)
    def solve(self, obs: KeypointObservations, gate_px=80.0,
              sigma_prior_m=0.10, sigma_vel_m=0.01, huber_px=10.0,
              iters=1500, lr=5e-4, fit_mask=None, verbose=True) -> RefineResult
        """fit_mask[T] bool: True のフレームの観測だけを当てはめに使う
        （ホールドアウト評価用。平滑項は全フレームに掛かる）"""

def refine_take(take_dir, keypoints=None, sigma_vel_m=0.003, gate_px=80.0,
                skip_overlay=False) -> dict
    """テイク一式の高水準 API（CLI / Pipeline 共用）。world_trajectory.npz を更新。
    前提: <take>/rgb/, <take>/intrinsics.json, <take>/world_trajectory.npz,
          <take>/keypoints_2d.npz"""

def refine_npz(base_npz, result: RefineResult, out_npz=None, report_json=None) -> str
def run_selftest() -> int            # 関門④a（面内復元 ≥90% 等）
```

> **注意**: `solve` の既定 `sigma_vel_m=0.01` に対し、`refine_take` は **0.003** を渡す。
> 実運用値は 0.003（関門④c でジャーク +3.5% を確認して採用）。

### 検出側 `tools/detect_keypoints_2d.py`（隔離 venv `.venv_mp`・handtraj 非依存）
出力 npz: `k2d[T,2,21,2]`（NaN=未検出）／`detected[T,2]`／`score[T,2]`／`width`／`height`／`n_frames`

---

## `handtraj/pipeline.py`

```python
@dataclass
class PipelineConfig:
    take: str
    video: str | None = None
    intrinsics: str | None = None
    force: bool = False
    skip_overlay: bool = False
    skip_vis3d: bool = False
    no_rectify: bool = False
    refine: bool = False          # 2D キーポイントによる並進リファイン（要 .venv_mp）

class Pipeline:
    def __init__(self, cfg: PipelineConfig)
    def ingest(self)     # 動画/intrinsics 配置 + 解像度調整
    def rectify(self)    # needs_rectify なら VideoRectifier（.rectified マーカー管理込み）
    def infer(self)      # tools/hawor_infer.py を subprocess で（プロセス分離維持）
    def overlay(self)    # OverlayRenderer（intrinsics 無ければ SLAM 由来の擬似値を生成）
    def export(self)     # TrajectoryExporter
    def refine(self)     # 検出（.venv_mp サブプロセス）→ refine_take
    def vis3d(self)      # Skeleton3DRenderer
    def run(self)        # ingest→rectify→infer→overlay→export→refine→vis3d
```

各段のスキップ判定条件は [ARCHITECTURE.md](ARCHITECTURE.md) §1 を参照。

---

## CLI（すべて薄いラッパ・引数は [SOURCE_INDEX.md](SOURCE_INDEX.md) の表を参照）

`scripts/run_pipeline.py` `scripts/calibrate_camera.py` `scripts/export_trajectory.py`
`scripts/refine_trajectory.py` `scripts/record_rgb.py`
`tools/overlay_check.py` `tools/visualize_3d.py` `tools/rectify_video.py`

### 互換シム（新規コードでは使わない）
- `adapters/load_hawor_frames.py` — `load_hawor_frames` / `load_world_joints` /
  `load_hawor_outputs` / `HAWOR_ROOT` / `FACES_NEW` / `HAND_ORDER` を handtraj から再エクスポート。
  `legacy/m3_surface_calibrate.py` の import を壊さないために存在
- `tools/video_util.py` — `from handtraj.video_io import Mp4Writer` の再エクスポート。参照ゼロ
