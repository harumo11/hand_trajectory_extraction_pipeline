# human_hand_video_to_vla_dataset

RGB 単眼動画から**両手・指（各21関節×2手）の world 座標軌道**を推定するパイプラインです。
[HaWoR](https://github.com/ThunderVVV/HaWoR)（CVPR 2025）をベースに、カメラキャリブレーションから
推定実行・検証用可視化・npz 形式でのエクスポートまでを一括で行います。

## 特徴

- mp4 動画を入力すると、world 座標の両手 21 関節軌道（npz）を出力
- チェスボードによるカメラキャリブレーション（盤面パターンの自動検出付き）
- 検証用の可視化を自動生成（メッシュ投影オーバーレイ動画・3D スケルトン動画）
- Python ライブラリ（`handtraj` パッケージ）としても利用可能

## 動作環境

- Ubuntu 24.04 / Python 3.12
- NVIDIA GPU（開発・検証環境は RTX 5080 / CUDA 12.x）
- PyTorch 2.7.1（cu128）
- ffmpeg

## ディレクトリ構成

```
handtraj/    ライブラリ本体（API 仕様は handtraj/CONTRACT.md）
scripts/     コマンドラインスクリプト
tools/       補助ツール（HaWoR 推論ドライバ・可視化・MANO 変換など）
adapters/    HaWoR 出力アダプタ（互換用の再エクスポート）
legacy/      旧 RealSense D405（深度カメラ）版のスクリプト（現在未使用）
patches/     HaWoR 側ソースへの必須パッチと適用スクリプト
third_party/ HaWoR のクローン先（リポジトリには含まれません）
captures/    録画・推定結果の置き場（リポジトリには含まれません）
```

## セットアップ

```bash
python3.12 -m venv .venv && source .venv/bin/activate

# 1) 依存パッケージ
pip install -r requirements.txt
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_hawor_cu128.txt
pip install pytorch-lightning==2.2.4 --no-deps && pip install lightning-utilities torchmetrics==1.4.0
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# 2) HaWoR の取得とパッチ適用（パッチ内容は patches/apply_patches.sh のコメント参照）
git clone --recursive https://github.com/ThunderVVV/HaWoR.git third_party/HaWoR
bash patches/apply_patches.sh

# 3) CUDA 拡張のビルド
export TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda-12   # GPU に合わせて変更
cd third_party/HaWoR/thirdparty/DROID-SLAM && python setup.py install && cd -
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# 4) HaWoR の学習済み重みを third_party/HaWoR/ 配下に配置（HaWoR の README 参照）
#    droid.pth / Metric3D / WiLoR detector.pt / hawor.ckpt / infiller.pt

# 5) MANO モデル（https://mano.is.tue.mpg.de で要アカウント登録）
python tools/prepare_mano.py --mano_zip ~/Downloads/mano_v1_2.zip
#    ※ 公式 pkl は直接配置せず必ずこのツールで変換してください（Python 3.12 では読めないため）
```

## 使い方

```bash
# 0) 一度だけ（推奨）: カメラキャリブレーション
#    チェスボードをいろいろな角度・距離で 20 秒ほど撮影した動画を用意して:
python scripts/calibrate_camera.py --video calib.mp4 --out intrinsics.json

# 1) 録画（UVC/Web カメラ）。既存の mp4 をそのまま使うこともできます
python scripts/record_rgb.py --out captures/take01_raw.mp4 --seconds 20 --preview

# 2) 推定パイプラインの実行
python scripts/run_pipeline.py --video captures/take01_raw.mp4 --take captures/take01 \
                               --intrinsics intrinsics.json
```

実行が終わると `captures/take01/` に以下が生成されます。

| ファイル | 内容 |
|---|---|
| `world_trajectory.npz` | 最終出力（下記「出力形式」参照） |
| `overlay/overlay.mp4` ほか | メッシュ投影オーバーレイ（推定品質の目視確認用） |
| `vis3d/world_3d.mp4` | 3D 可視化（カメラ軌跡 + 両手スケルトン） |

- `--intrinsics` を省略しても動作しますが、焦点距離が既定値（600px）にフォールバックし精度が落ちます。
- オプション: `--skip_overlay` / `--skip_vis3d`（可視化の省略）、`--force`（再実行）、
  `--no_rectify`（歪み補正・主点センタリングの前処理を無効化）

### ライブラリとして使う

```python
from handtraj import HaworSequence, TrajectoryExporter

seq = HaworSequence("captures/take01/rgb")     # HaWoR 出力へのアクセス
print(seq.joints_world.shape)                  # (T, 2, 21, 3)
TrajectoryExporter(seq).export("out.npz")
```

主要クラス: `Intrinsics` / `CameraCalibrator` / `VideoRectifier`（カメラ較正）、
`HaworSequence`（推定結果アクセス）、`TrajectoryExporter`（npz 出力）、
`OverlayRenderer` / `Skeleton3DRenderer`（可視化）、`Pipeline`（一括実行）。
詳細は `handtraj/CONTRACT.md` を参照してください。

## 出力形式（world_trajectory.npz）

| キー | 形状 | 内容 |
|---|---|---|
| `joints` | `[T, 2, 21, 3]` float32 | world 座標の手関節位置 [m]。軸1は [左手, 右手]。検出失敗フレームは NaN |
| `alpha` | スカラー | 適用済みグローバルスケール係数（既定 1.0） |
| `valid` | `[T]` bool | いずれかの手が有効なフレーム |
| `valid_per_hand` | `[T, 2]` bool | 手ごとの有効フラグ |

関節順序は OpenPose 互換（0=手首、1–4=親指、5–8=人差し指、9–12=中指、13–16=薬指、17–20=小指）。

## 精度に関する注意

- **スケールは近似メートル**です。実寸スケールは HaWoR 内蔵の単眼深度推定（Metric3D）に依存し、
  誤差 5〜15% 程度を見込んでください。較正値があれば
  `scripts/export_trajectory.py --alpha <値>` で差し替えられます。
- 画像面での手位置の整合は 1〜2cm 程度（モデル性能由来）。
- カメラの位置・回転ドリフトは未補正です。長時間の撮影では絶対位置の精度が低下します。

## ライセンスに関する注意

本リポジトリのコードは HaWoR（**CC BY-NC-ND 4.0**・非商用）と MANO（要登録・再配布不可）に
依存します。これらのモデル・重みはリポジトリに含まれないため、上記セットアップ手順に従って
各配布元のライセンスに同意のうえ取得してください。
