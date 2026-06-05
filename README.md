# 両手・指 world 軌道推定システム（RGB 単眼版 / HaWoR）

仕様の正本は `CLAUDE.md`（v2: RGB 単眼版）。本 README は実行手順と環境構築の記録。
旧 D405 パス（深度による面-面スケール較正）は休止中 — CLAUDE.md 付録A参照。

## ディレクトリ構成

```
handtraj/    ライブラリ本体（クラス中心。API 契約は handtraj/CONTRACT.md）
scripts/     CLI エントリポイント（run_pipeline / record_rgb / calibrate_camera / export_trajectory）
tools/       補助ツール（hawor_infer / overlay_check / visualize_3d / rectify_video / prepare_mano）
adapters/    CLAUDE.md §5 の一本化アダプタ（handtraj への互換シム）
legacy/      休止中の D405 深度較正パス（CLAUDE.md 付録A）
patches/     HaWoR vendored コードへの必須パッチ + 適用スクリプト
third_party/ HaWoR クローン先（git 管理外・README 手順で取得）
captures/    録画・推論成果物（git 管理外）
```

## パッケージ構成（2026-06 リファクタ済み）

ロジックは `handtraj/` パッケージにクラスとして集約。`scripts/` の CLI はすべて互換維持のシンウラッパ。
API 契約と移設対応は `handtraj/CONTRACT.md` を参照。

| モジュール | 主要クラス |
|---|---|
| `handtraj/camera.py` | `Intrinsics` / `project()` / `VideoRectifier` / `CameraCalibrator` |
| `handtraj/video_io.py` | `Mp4Writer`（H.264 自動再エンコード）/ 動画 I/O ヘルパ |
| `handtraj/hawor_adapter.py` | `HaworSequence`（HaWoR 出力への高水準アクセス・遅延キャッシュ）+ `HAND_BONES` 等 |
| `handtraj/export.py` | `TrajectoryExporter`（×α → npz + 手長診断） |
| `handtraj/visualization.py` | `OverlayRenderer` / `Skeleton3DRenderer` |
| `handtraj/pipeline.py` | `Pipeline` / `PipelineConfig`（全ステージのオーケストレーション） |

```python
# ライブラリとしての利用例
from handtraj import HaworSequence, TrajectoryExporter
seq = HaworSequence("captures/take03/rgb")
print(seq.joints_world.shape, seq.valid.sum())
TrajectoryExporter(seq).export("out.npz", alpha=1.0)
```

## パイプライン

```
record_rgb.py(UVC) or 任意mp4 ─► run_pipeline.py ┬ tools/hawor_infer.py   (HaWoR: MANO+SLAM+Metric3Dスケール)
calibrate_camera.py ─► intrinsics.json(推奨) ──┤ tools/overlay_check.py (関門③': 投影確認)
                                               └ export_trajectory.py   (×α, 既定α=1.0)
                                                  → <take>/world_trajectory.npz
```

## 使い方

```bash
source .venv/bin/activate

# 0) (一度だけ・推奨) カメラキャリブレーション — チェスボードをいろんな角度・距離で撮影して:
#    手持ちの 6x5マス・27mm ボードはパターン自動検出でそのまま使える（--cols/--rows 不要）
python scripts/calibrate_camera.py --video calib.mp4 --out intrinsics.json
#    ※ 小さめの盤（6x5マス）は1枚あたりの拘束が少ないため、20枚以上検出されるよう
#      長め（20秒程度）に多様な角度で撮るのを推奨

# 1) 録画（UVC/Webカメラ）— または任意の mp4 を用意
python scripts/record_rgb.py --out captures/take01_raw.mp4 --seconds 20 --preview

# 2) 一気通貫
python scripts/run_pipeline.py --video captures/take01_raw.mp4 --take captures/take01 \
                       --intrinsics intrinsics.json

# 最終出力: captures/take01/world_trajectory.npz
#   joints[T,2,21,3] (m, [left,right], invalid手はNaN) / alpha(=1.0) / valid[T] / valid_per_hand[T,2]
# 確認: captures/take01/overlay/*.png（静止画）+ overlay/overlay.mp4（全フレーム重畳動画・H.264）、
#       vis3d/world_3d.mp4（world座標3D可視化: カメラ軌跡+フラスタム+両手21関節スケルトン）、
#       export ログの手長チェック（16-20cm目安）
# 省略オプション: --skip_overlay（重畳系）/ --skip_vis3d（3D可視化）/ overlay_check.py --no_video
# 3D可視化の単体実行（視点・間引き変更可）:
#   python tools/visualize_3d.py --hawor <take>/rgb --trajectory <take>/world_trajectory.npz \
#       --out <take>/vis3d/world_3d.mp4 --elev 20 --azim -60 [--stride 2] [--no_flip]
```

- `--intrinsics` 省略時も動くが、HaWoR が焦点距離 600px 仮定にフォールバックし精度低下（警告表示）。
- スケールは HaWoR 内蔵（Metric3D）で**近似メートル**（誤差 5〜15% 想定）。実寸が必要なら
  `export_trajectory.py --alpha <較正値>` で差し替え可能（×α 機構は温存）。

## セットアップ（実施済み記録: RTX 5080 / sm_120 / Ubuntu 24.04 / Python 3.12）

1. `pip install -r requirements.txt`（軽量系）
2. `pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128`
3. `pip install -r requirements_hawor_cu128.txt`（HaWoR 依存。mmcv は 1.7.2 lite で代替、詳細はファイル内コメント）
4. `pip install pytorch-lightning==2.2.4 --no-deps && pip install lightning-utilities torchmetrics==1.4.0`
5. `pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.7.0+cu128.html`
6. HaWoR クローン + 必須パッチ + DROID-SLAM 拡張ビルド:
   ```bash
   git clone --recursive https://github.com/ThunderVVV/HaWoR.git third_party/HaWoR
   bash patches/apply_patches.sh    # sm_120 / torch2.x 対応（下表）。クローン直後に一度だけ
   export TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda-12
   cd third_party/HaWoR/thirdparty/DROID-SLAM && python setup.py install
   ```
7. pytorch3d ソースビルド（同上の env で `pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"`）
8. 重み: HF 系は wget、droid.pth / Metric3D は gdown で取得済み（配置先は HaWoR README 準拠）
9. **MANO（要ユーザー登録）**: https://mano.is.tue.mpg.de から `mano_v1_2.zip` を取得後:
   ```bash
   python tools/prepare_mano.py --mano_zip ~/Downloads/mano_v1_2.zip
   ```
   （公式 pkl は chumpy 依存で Python3.12 では読めないため、純 numpy へ変換して配置。直接配置禁止）

### vendored コードへの最小パッチ（HaWoR アルゴリズムは無改造）

パッチ実体は `patches/*.patch`（`bash patches/apply_patches.sh` で適用）。
`third_party/` は git 管理外（巨大・CC-BY-NC-ND/MANO ライセンスのため再配布しない）。

| ファイル | 変更 | 理由 |
|---|---|---|
| `third_party/HaWoR/thirdparty/DROID-SLAM/setup.py` | ハードコードの `-gencode sm_60..86` を削除 | `TORCH_CUDA_ARCH_LIST=12.0` を有効化（無いと RTX 5080 で kernel image エラー） |
| 同 `src/correlation_kernels.cu` / `src/altcorr_kernel.cu` | `AT_DISPATCH...(x.type(),` → `x.scalar_type()` | torch 2.x で旧 API が廃止 |
| 同 `thirdparty/lietorch/lietorch/src/lietorch_{gpu.cu,cpu.cpp}` | `DISPATCH...(group_id, x.type(),` → `x.scalar_type()` | 同上 |

※ torch≥2.6 の `torch.load` weights_only 問題は `tools/hawor_infer.py` 内のプロセス限定シムで対応（vendored 改変なし）。

## 関門の状態

- v1 関門①（sm_120 import）: **PASS**
- v1 関門②（面-面 selftest）: **PASS**（D405 休止パス。α=0.85 復元）
- v1 関門③（投影オーバーレイ）: **PASS** — カメラ規約 OpenCV 系で確定
- **v2 関門②'**（calibrate_camera --selftest）: **PASS** — 合成チェスボードで fx 誤差 0.01%
- **v2 関門③'**（実測 intrinsics での投影一致）: **PASS** — DJI Action2 実録画で確認
- **v2 最終**（一気通貫 + 手長妥当性）: **PASS** — example + 実録画の両方で確認（手長 17.4〜18.4cm）

## 休止中の D405 パス

`legacy/{record_d405,split_bag,m3_surface_calibrate}.py` として削除せず保持。
D405 復活時は v1 手順で α を算出し `scripts/export_trajectory.py --m3_report` に渡すだけ（CLAUDE.md 付録A）。
