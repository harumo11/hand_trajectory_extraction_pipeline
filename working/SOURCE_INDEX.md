# ソースインデックス

全 Python ファイル（`third_party/` 除く 25 本・約 3,100 行）の逐一索引。
**ファイルを追加・削除・改名したら本ファイルを更新すること。**

関連: [ARCHITECTURE.md](ARCHITECTURE.md)（実行フロー）・[API.md](API.md)（型・シグネチャ仕様）

---

## 逆引き: やりたいこと → 見るべきファイル

| やりたいこと | 見るファイル |
|---|---|
| 軌道の数値そのものを変えたい（スケール・補正） | `handtraj/export.py`, `handtraj/refine.py` |
| HaWoR 出力（.pth / SLAM npz）の読み方を知りたい | `handtraj/hawor_adapter.py` + [API.md](API.md) |
| 実行順・各段のスキップ条件・生成物の場所 | `handtraj/pipeline.py` + [ARCHITECTURE.md](ARCHITECTURE.md) |
| 投影・カメラ規約・歪み補正・キャリブレーション | `handtraj/camera.py`（OpenCV 系 +Z 前方・y 下） |
| オーバーレイ画像 / 3D スケルトン動画の描画 | `handtraj/visualization.py` |
| mp4 の読み書き・fps/解像度取得 | `handtraj/video_io.py` |
| 出力 npz のキー・形式 | `handtraj/export.py::TrajectoryExporter.export` + `handtraj/refine.py::refine_npz` |
| HaWoR 本体をどう呼んでいるか | `tools/hawor_infer.py`（別プロセス・HaWoR root に chdir） |
| MediaPipe による 2D 検出 | `tools/detect_keypoints_2d.py`（**隔離 venv `.venv_mp` 専用**） |
| CLI の引数を足したい | `scripts/*.py`（薄いラッパ。ロジックは handtraj 側） |
| 精度の実測値・誤差要因 | [ACCURACY.md](ACCURACY.md) |
| 過去の経緯・関門記録・D405 復活手順 | [HISTORY.md](HISTORY.md) |

## 触る前に知っておくこと

- **新規コードは `handtraj/` に書く。** `scripts/` `tools/` は argparse だけの薄いラッパにする。
- `tools/video_util.py` は **孤児シム** — リポジトリ内の誰も import していない。
  新規コードは `handtraj.video_io` を直接使う。
- `adapters/load_hawor_frames.py` は旧 API 互換シム。生きている参照は
  `legacy/m3_surface_calibrate.py` からの遅延 import 1 箇所のみ。新規コードでは使わない。
- `handtraj/__init__.py` は全サブモジュールを eager import するが、
  **torch は各モジュール内で遅延 import** している。`import handtraj` で torch/CUDA が
  走らない設計を壊さないこと。
- `print` の文言（`[INFO]` / `[WARN]` / `[CHECK]` / `[RUN]` / `[DONE]` / `[ERROR]`）は
  ログ互換のため既存表現を維持する。

---

## handtraj/ — ライブラリ本体

| ファイル | 行 | 目的 | 主なエントリポイント | リポジトリ内 import | 環境要件 |
|---|---:|---|---|---|---|
| `__init__.py` | 38 | 主要 14 名の再エクスポート | `__all__` | 全サブモジュール（eager） | なし |
| `hawor_adapter.py` | 277 | HaWoR `seq_folder` 出力の高水準リーダ。MANO forward + SLAM カメラを初回アクセス時に 1 度だけ計算しキャッシュ | `HaworSequence(seq_folder)` / 定数 `HAWOR_ROOT` `HAND_ORDER` `HAND_BONES` `FACES_NEW` `WRIST` `MIDDLE_MCP` `MIDDLE_TIP` | なし（リーフ） | **GPU・torch・MANO・`third_party/HaWoR`**（`_load()` 内で遅延 import） |
| `pipeline.py` | 245 | 全段の実行順とスキップ判定 | `PipelineConfig` / `Pipeline(cfg)` の `.ingest/.rectify/.infer/.overlay/.export/.refine/.vis3d/.run` | `camera` `export` `hawor_adapter` `video_io` `visualization`／遅延 `refine`／subprocess `tools/hawor_infer.py` `tools/detect_keypoints_2d.py` | 実行時に GPU |
| `camera.py` | 311 | 内部パラメータ・投影・歪み補正・チェスボード較正 | `Intrinsics` / `project(pts_cam, K)` / `VideoRectifier` / `CameraCalibrator` | 遅延 `video_io.Mp4Writer` | cv2 のみ（ヘッドレス可） |
| `export.py` | 130 | `world_joints × α` → npz ＋ 手長サニティチェック | `TrajectoryExporter(seq).export(...)` / `hand_length_stats(joints, valid)` | `hawor_adapter`（関節インデックス定数） | `HaworSequence` 経由で GPU |
| `refine.py` | 328 | 2D キーポイントによるフレーム別・手別の並進リファイン（torch Adam） | `KeypointObservations` / `TranslationRefiner` / `RefineResult` / `refine_npz` / `refine_take` / `run_selftest` ／ CLI は `--selftest` のみ | 遅延 `hawor_adapter` `camera` `visualization` | torch（CUDA 任意）。`refine_take` は `.venv_mp` 生成の npz が前提 |
| `visualization.py` | 212 | メッシュ投影オーバーレイ／3D world スケルトン | `OverlayRenderer(seq, intr)` / `Skeleton3DRenderer(...)` / `equal_limits` | `camera.project` `hawor_adapter.HAND_BONES` `video_io.Mp4Writer` | matplotlib（`Agg` 強制）・ffmpeg |
| `video_io.py` | 81 | mp4 I/O | `Mp4Writer` / `video_fps` / `video_size` / `read_frame` / `iter_frames` | なし（リーフ） | cv2・ffmpeg |

## scripts/ — CLI

| ファイル | 行 | 目的 | CLI 引数 | 内部 import |
|---|---:|---|---|---|
| `run_pipeline.py` | 65 | **主 CLI**。取込〜出力まで一括 | `--video --take --intrinsics --force --skip_overlay --skip_vis3d --no_rectify --refine` | `handtraj.pipeline` |
| `calibrate_camera.py` | 156 | チェスボード → `intrinsics.json`。`--selftest` は合成盤で fx 誤差検証 | `--selftest --video --images --out --cols --rows --square_mm --max_frames` | `handtraj.camera.CameraCalibrator` |
| `export_trajectory.py` | 56 | エクスポート段だけ再実行（α 差し替え用） | `--hawor --out --alpha --m3_report --fps` | `handtraj.hawor_adapter` `handtraj.export` |
| `refine_trajectory.py` | 47 | 実行済みテイクへのリファイン後がけ | `--take --keypoints --sigma_vel --gate_px --skip_overlay` | `handtraj.refine.refine_take` |
| `record_rgb.py` | 90 | UVC/Web カメラ録画（実効 fps 計測・警告付き） | `--out --device --size --fps --seconds --preview` | **なし**（cv2 のみ。意図的に単独動作） |

## tools/ — 単機能 CLI・環境構築

| ファイル | 行 | 目的 | CLI 引数 | 内部 import | 環境要件 |
|---|---:|---|---|---|---|
| `hawor_infer.py` | 94 | HaWoR 推論ドライバ。`demo.py` の 4 呼び出しを可視化依存なしで再現。`torch.load(weights_only=False)` を monkeypatch し HaWoR root に chdir | `--video_path --intrinsics --img_focal --checkpoint --infiller_weight` | **なし**（プロセス分離のため意図的） | **GPU・全重み・patches 適用済み `third_party/HaWoR`** |
| `detect_keypoints_2d.py` | 112 | MediaPipe HandLandmarker → `keypoints_2d.npz` | `--video --out --min_conf` | **なし**（`.venv_mp` で動くため） | **隔離 venv `.venv_mp`**（mediapipe は numpy≥2 を要求し主 venv と非互換）。初回に `hand_landmarker.task` を `~/.cache/handtraj` へ DL |
| `overlay_check.py` | 61 | 投影オーバーレイ CLI（関門③'） | `--hawor --video --intrinsics --out --num --video_out --no_video` | `handtraj.camera` `.hawor_adapter` `.visualization` | GPU |
| `visualize_3d.py` | 65 | 3D world スケルトン動画 CLI | `--hawor --trajectory --out --fps --stride --size --elev --azim --no_flip` | `handtraj.hawor_adapter` `.visualization` | GPU |
| `rectify_video.py` | 45 | 歪み補正を単体実行 | `--video --intrinsics --out_video --out_intrinsics` | `handtraj.camera` | cv2 のみ |
| `prepare_mano.py` | 158 | 公式 MANO pkl（chumpy 依存）→ 純 numpy 変換して `third_party/HaWoR/_DATA/` へ設置。smplx で検証 | `--mano_zip --mano_dir --skip_verify` | なし | 要手動取得 `mano_v1_2.zip`。検証に torch + smplx |
| `video_util.py` | 15 | `Mp4Writer` の後方互換再エクスポート | — | `handtraj.video_io` | **孤児（参照ゼロ）** |
| `setup_keypoint_env.sh` | 21 | `.venv_mp` 構築（python3.12 + mediapipe + opencv）。冪等 | — | — | — |

## adapters/ — 旧 API 互換シム

| ファイル | 行 | 内容 |
|---|---:|---|
| `__init__.py` | 0 | パッケージマーカー（legacy の遅延 import 用に必要） |
| `load_hawor_frames.py` | 76 | `HaworSequence` の上に旧関数 API を被せたシム。`load_hawor_outputs` / `load_hawor_frames` / `load_world_joints` ＋ `HAND_ORDER` `HAWOR_ROOT` `FACES_NEW` を再エクスポート。CLI なし |

## legacy/ — D405 深度カメラ時代（休止・[HISTORY.md](HISTORY.md) 参照）

| ファイル | 行 | 目的 | CLI 引数 | 環境要件 |
|---|---:|---|---|---|
| `m3_surface_calibrate.py` | 227 | 面-面レンダリングによるグローバル α 較正（純 numpy デプスラスタライザ）。`--selftest` あり | `--selftest --hawor --depth_dir --intrinsics --out` | GPU（`adapters` 経由・遅延 import） |
| `split_bag.py` | 121 | RealSense `.bag` → `rgb.mp4` + 16bit 深度 PNG + `intrinsics.json` | `--bag --out` | `pyrealsense2` |
| `record_d405.py` | 79 | D405 の RGB+Depth を `.bag` 録画 | `--out --seconds --preview` | `pyrealsense2` ＋ 実機 D405 |

## patches/ — third_party/HaWoR への必須パッチ

| ファイル | 内容 |
|---|---|
| `01_droid_slam_sm120_torch2x.patch` | DROID-SLAM `setup.py` の `-gencode sm_60..86` ハードコード削除（`TORCH_CUDA_ARCH_LIST=12.0` を有効化）＋ CUDA カーネルの `.type()` → `.scalar_type()` |
| `02_lietorch_torch2x.patch` | lietorch の DISPATCH マクロ `.type()` → `.scalar_type()` |
| `apply_patches.sh` | 冪等な適用スクリプト（`git apply --reverse --check` で適用済み検知） |

## テストの所在

専用の `tests/` ディレクトリはない。自己検証は 3 つの `--selftest` のみ:

- `python scripts/calibrate_camera.py --selftest` — 合成チェスボードで fx 誤差 <1%（関門②'）
- `python -m handtraj.refine --selftest` — 人工オフセットの復元率（関門④a）
- `python legacy/m3_surface_calibrate.py --selftest` — 面-面 α 復元（休止パス）
