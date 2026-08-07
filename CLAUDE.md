# CLAUDE.md — 両手・指 world 軌道推定システム（RGB 単眼）

RGB 単眼動画（頭部装着等）から、**両手・指（各 21 関節 × 2 手）の world 座標軌道**を推定する。
ベースは **HaWoR**。SLAM・カメラ↔world 合成・メトリックスケール推定は HaWoR が内部で行う。
ユーザー向けのセットアップ・使い方は `README.md`。

**用語**: **α**＝world 軌道に掛けるグローバルスケール係数（**本版は 1.0 固定**）／
**MANO**＝手のパラメトリックモデル（778 頂点・21 関節）／
**Metric3D**＝HaWoR が SLAM スケール決定に内部使用する単眼メトリック深度モデル。

## 方針（厳守）

- **HaWoR 本体は改造しない。** vendored への変更はビルド/torch 互換の最小パッチのみ（`patches/` に記録）。
- **実寸スケールは HaWoR 内蔵（Metric3D）を信頼し α=1.0 固定。**
  ただし出力 npz とエクスポート段の ×α 機構は温存する（手実寸較正・マーカー較正・D405 復活の
  いずれでも α の差し替えだけで実寸化できるように）。
- 内部パラメータは**チェスボードキャリブレーションで実測**して HaWoR に渡す
  （未指定だと HaWoR は焦点距離 600px にフォールバックし精度が落ちる）。
- 入力動画は intrinsics に基づき**理想ピンホール化してから** HaWoR に渡す
  （`run_pipeline.py` が自動実施、`--no_rectify` で無効）。
  **これは衛生措置であって精度改善策ではない** — 実測で改善は確認されていない（`working/ACCURACY.md` §3）。
- **既存実装を流用し、新規実装は最小限。新規コードは `handtraj/` に書く**
  （`scripts/` `tools/` は argparse だけの薄いラッパに保つ）。
- **やらない**: HaWoR 内部改変／時間変化 α・再積分／位置ドリフト補正／独自 SLAM／独自スケール推定。

## 全体フロー

```
UVCカメラ ─ scripts/record_rgb.py ─► rgb.mp4 ─┐
任意の mp4 ──────────────────────────────────┤
チェスボード動画 ─ scripts/calibrate_camera.py ─► intrinsics.json（任意だが強く推奨）
                                      ▼
        tools/hawor_infer.py  [別プロセス] 検出→MANO→masked DROID-SLAM＋Metric3Dスケール→world合成
                                      ▼
        handtraj.HaworSequence ─► overlay（投影一致の目視確認）
                                      ▼
        world_true = α · world_joints            （本版は α=1.0）
                                      ▼
        任意: 2Dキーポイント並進リファイン（--refine。要 .venv_mp）
                                      ▼
        world_trajectory.npz — joints[T,2,21,3](m), alpha, valid[T], valid_per_hand[T,2]
```

各段のスキップ条件・生成物・テイクディレクトリの中身は `working/ARCHITECTURE.md`。

## ソースツリー

```
handtraj/            ライブラリ本体（新規コードはここ）
  hawor_adapter.py   HaWoR 出力の読み口。HaworSequence（joints_world / verts_cam / valid / R_c2w …）
  pipeline.py        実行順とスキップ判定（ingest→rectify→infer→overlay→export→refine→vis3d）
  camera.py          Intrinsics / project / VideoRectifier / CameraCalibrator
  export.py          world_joints × α → npz ＋ 手長サニティチェック
  refine.py          2D キーポイント並進リファイン（torch Adam）
  visualization.py   OverlayRenderer / Skeleton3DRenderer
  video_io.py        Mp4Writer / video_fps / video_size / iter_frames
scripts/             CLI: run_pipeline（主）/ calibrate_camera / export_trajectory
                          / refine_trajectory / record_rgb
tools/               単機能 CLI: hawor_infer[別プロセス] / detect_keypoints_2d[.venv_mp]
                          / overlay_check / visualize_3d / rectify_video / prepare_mano
adapters/            旧 API 互換シム（legacy 用。新規コードでは使わない）
legacy/              D405 時代（休止。working/HISTORY.md §4）
patches/             third_party/HaWoR への必須パッチ
working/             詳細ドキュメント（下記）
```

**ファイルを追加・削除・改名したら `working/SOURCE_INDEX.md` を更新すること。**

## インターフェース

- **入力**: `<take>/rgb.mp4` ＋ `<take>/intrinsics.json`（任意・推奨）
  ＝ `{width, height, fx, fy, cx, cy, model, coeffs, fps}`
- **カメラ規約は OpenCV 系**（+Z 前方・y 下・`u = fx·X/Z + cx`）。実証済み。
- **出力 `world_trajectory.npz`**:

  | キー | 型 | 内容 |
  |---|---|---|
  | `joints` | `[T,2,21,3]` | world 手関節（m）。`[left, right]`。invalid は NaN |
  | `alpha` | float | グローバルスケール係数（現行 1.0） |
  | `valid` | `[T]` bool | どちらかの手が有効 |
  | `valid_per_hand` | `[T,2]` bool | 手ごとの有効フラグ |
  | `refined` / `delta_t_world` | bool / `[T,2,3]` | `--refine` 適用時のみ追加 |

- API シグネチャの完全版は `working/API.md`。

## 環境の地雷

RTX 5080 (sm_120) / Ubuntu 24.04 / Python 3.12 / torch 2.7.1+cu128。構築手順は `README.md`。

- **numpy<2 必須。** mediapipe を主 venv に入れると numpy が 2.x に上がって環境が壊れる。
  2D 検出は隔離 venv `.venv_mp`（`bash tools/setup_keypoint_env.sh`）で動かす。
- **MANO は `tools/prepare_mano.py` 経由で配置**（chumpy 除去変換が必要。公式 pkl の直接配置は不可）。
- 再クローン時は `bash patches/apply_patches.sh`（適用済み検知付き）。
- `import handtraj` で torch/CUDA が走らない設計（torch は各モジュールで遅延 import）。壊さないこと。
- `print` の `[INFO]`/`[WARN]`/`[CHECK]`/`[RUN]`/`[DONE]` 文言はログ互換のため維持する。

## 既知の限界

- **実寸精度は Metric3D 依存（誤差 5〜15% 想定）。** サニティチェックは手長 16〜20cm（実測 17.4〜18.4cm）。
- **画像面の位置整合は素で 25〜55px、`--refine` で 16px。** 残る 16px は MANO 関節中心と
  MediaPipe ランドマークの**定義差**であり軌道誤差ではない。奥行き成分の補正は 2D 観測の
  情報限界により部分的（面内 96% / 奥行き 65%）。
- 位置・回転ドリフトは未補正（HaWoR 任せ）。長尺の絶対精度は静止区間で要実測。
- 手の検出失敗・画面外で `valid=False`（NaN）。HaWoR は 30fps 前提。
- 数値の実測条件と誤差要因の詳細は `working/ACCURACY.md`。
  **精度改善を主張するときは必ず定量比較とセットにすること。**

## 詳細ドキュメント（working/）

| ファイル | 内容 |
|---|---|
| `working/SOURCE_INDEX.md` | 全ソースの逐一索引 ＋「やりたいこと → 見るファイル」逆引き表 |
| `working/ARCHITECTURE.md` | 実行フロー（各段のスキップ条件・生成物）・依存グラフ・テイクの構成 |
| `working/API.md` | handtraj の型・シグネチャ仕様・コーディング規約 |
| `working/ACCURACY.md` | 精度の実測値と誤差要因（数値を主張する前に読む） |
| `working/HISTORY.md` | v1→v2 の経緯・関門の記録・D405 パス復活手順・環境構築の記録 |
