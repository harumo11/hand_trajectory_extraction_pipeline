# アーキテクチャ — 実行フローとデータフロー

関連: [SOURCE_INDEX.md](SOURCE_INDEX.md)（ファイル索引）・[API.md](API.md)（シグネチャ仕様）

---

## 1. `scripts/run_pipeline.py` 実行時の呼び出し順

エントリ: `run_pipeline.main()` → argparse → `PipelineConfig(...)` → `Pipeline(cfg).run()`
（`handtraj/pipeline.py`）。以下 7 段を順に実行する。

### ① `Pipeline.ingest()`
- `--video` を `<take>/rgb.mp4` へ、`--intrinsics` を `<take>/intrinsics.json` へコピー
- **スキップ条件**: コピー先が既存かつ `--force` なし
- intrinsics の解像度が動画と異なれば `_adjust_intrinsics_to_video()` が
  `video_io.video_size` ＋ `Intrinsics.scaled_to` で比例スケールして上書き（事故防止の安全装置）
- `rgb.mp4` が無ければ assert 失敗

### ② `Pipeline.rectify()`
- **スキップ条件**: intrinsics なし ／ `--no_rectify` ／ `<take>/.rectified` マーカー存在（`--force` で無視）
- `Intrinsics.needs_rectify()` が True のときのみ実行:
  - 元を `rgb_raw.mp4` / `intrinsics_raw.json` へ退避
  - `camera.VideoRectifier.rectify_video()`（cv2 `initUndistortRectifyMap` + `remap`、
    書き出しは `video_io.Mp4Writer` — `close()` 時に ffmpeg で H.264 再エンコード）
  - `Intrinsics.rectified()` を `intrinsics.json` に保存
- False なら「補正不要」と表示。いずれも `.rectified` マーカーを作成
- intrinsics なしの場合は「HaWoR は焦点距離 600px 仮定」の `[WARN]`

### ③ `Pipeline.infer()` — **subprocess（プロセス分離）**
- **スキップ条件**: `<take>/rgb/world_space_res.pth` が既存（`--force` で削除して再実行）
- `sys.executable tools/hawor_infer.py --video_path <take>/rgb.mp4 [--intrinsics ...]`
- `hawor_infer.main()` の内部順序:
  1. intrinsics の `fx` を `img_focal` として読む
  2. `torch.load` を `weights_only=False` に monkeypatch（torch≥2.6 対策・vendored 無改変）
  3. `os.chdir(HAWOR_ROOT)`
  4. `detect_track_video` → `hawor_motion_estimation` → `hawor_slam` → `hawor_infiller`
- `hawor_slam` が masked DROID-SLAM を走らせ、Metric3D の単眼メトリック深度で軌道スケールを決定

### ④ `Pipeline.overlay()`
- **スキップ条件**: `--skip_overlay`
- intrinsics が無ければ `_make_pseudo_intrinsics()` が SLAM npz の `img_focal` / `img_center` と
  1 枚目の画像サイズから擬似 intrinsics を生成（`[WARN]`・投影確認専用）
- `HaworSequence(<take>/rgb)` を構築 → 初回 `_load()` で joblib ロード ＋
  `run_mano_left`/`run_mano` のチャンク実行 ＋ `load_slam_cam`、`verts_cam` まで計算
- `valid` が全滅なら `[ERROR]` を出して return
- `OverlayRenderer.snapshots(rgb.mp4, <take>/overlay, 6)` ＋ `.render_video(... overlay/overlay.mp4)`

### ⑤ `Pipeline.export()`
- **スキップ条件**: `world_trajectory.npz` が既存かつ `--force` なし
- fps は `intrinsics.fps` → 無ければ `video_io.video_fps(rgb.mp4)`
- `TrajectoryExporter(seq).export(<take>/world_trajectory.npz, fps=fps)`（α=1.0）
- 手長の `[CHECK]` 判定を print（`hand_length_stats`）

### ⑥ `Pipeline.refine()`
- **スキップ条件**: `--refine` 未指定 ／ npz が既に `refined=True`（`--force` で無視）
- `keypoints_2d.npz` が無ければ **subprocess**
  `.venv_mp/bin/python tools/detect_keypoints_2d.py --video rgb.mp4 --out keypoints_2d.npz`
  （`.venv_mp` が無ければ `[WARN]` を出してリファイン自体をスキップ）
- その後は**同一プロセス内**で `handtraj.refine.refine_take()`:
  `KeypointObservations.match_and_gate`（左右スワップ補正 ＋ 残差中央値ゲート）
  → `TranslationRefiner.solve`（torch Adam・Huber ＋ 事前分布 ＋ 平滑化）
  → `refine_npz` が `world_trajectory.npz` を更新（`refined=True`, `delta_t_world` 追加）
  ＋ `refine_report.json` ＋ `overlay_refined/`

### ⑦ `Pipeline.vis3d()`
- **スキップ条件**: `--skip_vis3d` ／ `vis3d/world_3d.mp4` が既存かつ `--force` なし
- `world_trajectory.npz` を**読み直す**ため、⑥ が先に走っている以上
  描画されるのは**リファイン後**の関節
- `seq.cam_centers * alpha` でカメラ位置を関節と同じ尺度に揃える
- `Skeleton3DRenderer(...).render_video(vis3d/world_3d.mp4, fps, stride=1, 1280x720, elev=20, azim=-60)`

最後に `[DONE]` 行を出して 0 を返す。

---

## 2. テイクディレクトリのレイアウト

```
<take>/
├─ rgb.mp4                       入力（補正後）。① で配置、② で上書きされうる
├─ rgb_raw.mp4                   ② の退避元（補正した場合のみ）
├─ intrinsics.json               補正後の内部パラメータ
├─ intrinsics_raw.json           ② の退避元
├─ .rectified                    ② 実行済みマーカー
├─ rgb/                          HaWoR の seq_folder（<video_dir>/<basename>/）
│   ├─ world_space_res.pth       ③ の主出力（MANO パラメータ・world 姿勢）
│   ├─ SLAM/hawor_slam_w_scale_<s>_<e>.npz   カメラ姿勢 + scale + img_focal + img_center
│   ├─ extracted_images/*.jpg
│   └─ intrinsics_pseudo.json    ④ で intrinsics 不在時のみ生成
├─ overlay/                      ④: overlay.mp4 + スナップショット PNG 6 枚
├─ world_trajectory.npz          ⑤ の最終出力（⑥ で上書き更新されうる）
├─ keypoints_2d.npz              ⑥: MediaPipe 2D 検出結果
├─ refine_report.json            ⑥ の診断
├─ overlay_refined/              ⑥ 後のオーバーレイ
└─ vis3d/world_3d.mp4            ⑦
```

---

## 3. 内部 import 依存グラフ

```
scripts/run_pipeline.py       ──► handtraj.pipeline
scripts/calibrate_camera.py   ──► handtraj.camera
scripts/export_trajectory.py  ──► handtraj.hawor_adapter, .export
scripts/refine_trajectory.py  ──► handtraj.refine
scripts/record_rgb.py         ──► （なし）

tools/overlay_check.py        ──► handtraj.camera, .hawor_adapter, .visualization
tools/visualize_3d.py         ──► handtraj.hawor_adapter, .visualization
tools/rectify_video.py        ──► handtraj.camera
tools/video_util.py           ──► handtraj.video_io          【孤児シム・参照ゼロ】
tools/hawor_infer.py          ──► （なし: third_party のみ）  【プロセス分離境界】
tools/detect_keypoints_2d.py  ──► （なし）                    【venv 分離境界 .venv_mp】
tools/prepare_mano.py         ──► （なし: third_party へ書く）

adapters/load_hawor_frames.py ──► handtraj.hawor_adapter      【互換シム】
legacy/m3_surface_calibrate.py ┈► adapters.load_hawor_frames  （遅延・関数内）
legacy/split_bag.py, record_d405.py ──► （なし）

handtraj/pipeline.py      ──► camera, export, hawor_adapter, video_io, visualization
                          ┈► refine                （遅延: Pipeline.refine() 内）
                          ⇢ subprocess: tools/hawor_infer.py, tools/detect_keypoints_2d.py
handtraj/visualization.py ──► camera, hawor_adapter, video_io
handtraj/export.py        ──► hawor_adapter
handtraj/refine.py        ┈► hawor_adapter, camera, visualization  （遅延: refine_take 内）
handtraj/camera.py        ┈► video_io               （遅延: rectify_video 内）
handtraj/hawor_adapter.py ──► （リーフ。third_party/HaWoR は _load() 内で遅延）
handtraj/video_io.py      ──► （リーフ）
handtraj/__init__.py      ──► 全サブモジュール（eager。ただし torch は各所で遅延）
```

- リーフ: `video_io`, `hawor_adapter`
- import 時の循環なし。`pipeline ↔ refine` と `refine → visualization` は呼び出し時まで遅延
- **2 つの分離境界**: HaWoR 推論（別プロセス・torch monkeypatch と chdir のため）と
  MediaPipe 検出（別 venv・numpy のメジャーバージョン衝突のため）

---

## 4. スケールの出どころ

HaWoR は masked DROID-SLAM の軌道スケールを Metric3D の単眼メトリック深度で推定する
（SLAM npz の `scale`）。よって world 出力は「近似メートル」。
本システムの α はその上に掛けるグローバル係数で、**現行は 1.0 固定**。
誤差の実測は [ACCURACY.md](ACCURACY.md)、α 較正の選択肢は同ファイルおよび [HISTORY.md](HISTORY.md)（D405 パス）を参照。
