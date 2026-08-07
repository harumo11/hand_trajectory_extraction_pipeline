# 開発経緯・検証記録

CLAUDE.md から切り出した経緯メモ。**日々の開発では読まなくてよい。**
「なぜこうなっているか」を辿るときと、D405 パスを復活させるときに参照する。

---

## 1. v1（D405 深度カメラ）→ v2（RGB 単眼）への転換

**2026-06-05**: RealSense D405 が使用不可になったため、深度カメラ前提の v1 から
**通常の RGB カメラ（UVC/Web カメラ または任意の mp4）のみ**で動くシステムへ転換した。

主な差分:

| 項目 | v1 | v2（現行） |
|---|---|---|
| 入力 | D405 の `.bag`（RGB + 整列深度） | 任意の mp4 |
| スケール決定 | 面-面レンダリングによる α 較正（実測深度と MANO メッシュを突き合わせ） | HaWoR 内蔵 Metric3D を信頼して **α=1.0 固定** |
| 較正コード | `legacy/m3_surface_calibrate.py` | なし（α の受け皿だけ温存） |

**旧パスは削除せず「休止」にしてある。** 理由は、実寸精度が要件化したときに
D405 を入手すれば α 較正の仕組みごと復活できるようにするため。
`export_trajectory.py --m3_report` というインターフェースもそのために残してある（§4）。

### 廃止した用語
かつて CLAUDE.md で使っていた以下の用語は、現行仕様には登場しない。
古いコミットメッセージや `legacy/` を読むときのための対訳:

- **M2'** — 取込段階（録画または既存 mp4 の受け入れ ＋ 内部パラメータ取得）
- **エクスポート段（旧 M3+）** — HaWoR 出力をアダプタ経由で npz に変換する段階
- **整列（alignment）／面-面／デプスレンダ** — いずれも v1 の深度較正まわりの用語

---

## 2. 実装タスクの完了履歴

この順に実装し、すべて完了している。

1. `scripts/calibrate_camera.py` — チェスボード動画/画像 → `intrinsics.json`（`--selftest` 付き）
2. `scripts/record_rgb.py` — UVC 録画 → mp4（実効 fps 計測・警告付き。HaWoR は 30fps 前提）
3. `scripts/export_trajectory.py` — `world_joints × α` → npz ＋ 診断
4. `scripts/run_pipeline.py` — mp4 取込 → HaWoR → overlay → エクスポート
5. 実カメラでの一気通貫（キャリブ → 録画 → 軌道出力）
6. 2D キーポイント並進リファイン — `handtraj/refine.py` ＋ `tools/detect_keypoints_2d.py`（隔離 venv）
   ＋ `scripts/refine_trajectory.py` / `run_pipeline --refine`（npz 追加キー: `refined` / `delta_t_world`）

**2026-06 リファクタ**: それまで `adapters/` `tools/` に散っていたロジックを
`handtraj/` パッケージへライブラリ化し、CLI は全て薄いラッパにした。
API 仕様は [API.md](API.md)（旧 `handtraj/CONTRACT.md`）。

---

## 3. 関門の記録（2026-06-05 時点・すべて PASS）

「通るまで先へ進まない」チェックポイントとして運用したもの。

| 関門 | 内容 | 結果 |
|---|---|---|
| v1 ① | sm_120 ビルド・import | **PASS** |
| v1 ② | 面-面 selftest | **PASS**（α=0.85 を復元・std 0.0001） |
| v1 ③ | 投影オーバーレイ | **PASS** — カメラ規約が **OpenCV 系**（+Z 前方・y 下・`u=fx·X/Z+cx`）と確定 |
| — | M3 統合検証（実メッシュ × 合成深度） | α=0.85 を誤差 **0.00%** で復元 |
| v2 ②' | `calibrate_camera.py --selftest` | **PASS** — fx 誤差 0.01〜0.39%（2 ケース） |
| v2 ③' | 実測 intrinsics での投影一致 | **PASS** — DJI Action2 実録画で確認 |
| v2 最終 | 一気通貫 ＋ 手長妥当性 | **PASS** — example + 実録画で手長 17.4〜18.4cm（目安 16〜20cm） |
| リファクタ回帰（2026-06） | take03 npz の数値一致・overlay PNG のバイト一致・契約準拠レビュー | **PASS** |
| ④a | リファイン selftest | **PASS** — 人工オフセット ±2cm を面内 95.7% 復元、残差 32.5→3.4px |
| ④b | ホールドアウト（非循環）評価 | **PASS** — 偶数フレーム学習 → 奇数フレームで 55.0→16.1px（fit 15.8px とほぼ同値＝過学習なし） |
| ④c | 副作用なし | **PASS** — 手長完全不変（剛体シフト）・ジャーク +3.5%（σ_vel=0.003 採用） |

④b の残差フロア 16px の正体は [ACCURACY.md](ACCURACY.md) を参照（軌道誤差ではなく関節定義差）。

---

## 4. 付録: 休止中の D405 パス（`legacy/` 配下）

### 構成
```
legacy/record_d405.py        D405 の RGB+Depth を .bag に録画
      ↓
legacy/split_bag.py          .bag → rgb.mp4 + 整列深度 PNG + intrinsics.json
      ↓
legacy/m3_surface_calibrate.py   面-面レンダリングでグローバル α を較正
      ↓
scripts/export_trajectory.py --m3_report <report>    ×α で実寸化
```

### 復活手順
1. D405 を入手
2. v1 どおり `record_d405.py` で録画 → `split_bag.py` で分離
3. `m3_surface_calibrate.py` で α を算出
4. `export_trajectory.py --m3_report <report>` に差し替える

**インターフェースは温存済み**のため、コード変更は不要。
`world_trajectory.npz` の `alpha` フィールドがそのための受け皿。

---

## 5. 環境構築の記録

RTX 5080 (sm_120) / Ubuntu 24.04 / Python 3.12 / torch 2.7.1+cu128。
手順は README.md、必須パッチは `patches/`。

### vendored コードへの最小パッチ（HaWoR アルゴリズムは無改造）

| ファイル | 変更 | 理由 |
|---|---|---|
| `third_party/HaWoR/thirdparty/DROID-SLAM/setup.py` | `-gencode sm_60..86` のハードコード削除 | `TORCH_CUDA_ARCH_LIST=12.0` を有効化（無いと RTX 5080 で kernel image エラー） |
| 同 `src/correlation_kernels.cu` / `src/altcorr_kernel.cu` | `AT_DISPATCH...(x.type(),` → `x.scalar_type()` | torch 2.x で旧 API 廃止 |
| 同 `thirdparty/lietorch/.../lietorch_{gpu.cu,cpu.cpp}` | `DISPATCH...(group_id, x.type(),` → `x.scalar_type()` | 同上 |

再クローン時は `bash patches/apply_patches.sh`（適用済み検知付き）。

### その他の記録
- torch≥2.6 の `torch.load` weights_only 問題は `tools/hawor_infer.py` 内の
  プロセス限定 monkeypatch で対応（vendored 改変なし）
- 重み取得: HF 系（`detector.pt` / `hawor.ckpt` / `infiller.pt` / `model_config.yaml`）は wget、
  `droid.pth` / Metric3D は gdown（Google Drive ID は HaWoR README のリンク）
- MANO は登録制のため `tools/prepare_mano.py` 経由で配置（chumpy 除去変換。直接配置は不可）
- **mediapipe を主 venv に pip で入れると numpy が 2.x に上がって環境が壊れる**（numpy<2 必須）。
  そのため `.venv_mp` に隔離している。うっかり壊した場合は
  `numpy==1.26.4` / `opencv-python==4.11.0.86` を再固定する
