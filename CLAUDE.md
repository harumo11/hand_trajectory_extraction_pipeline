# CLAUDE.md — 両手・指 world 軌道推定システム（RGB単眼版・単一仕様書）

> **v2 (2026-06-05)**: D405 が使用不可になったため、深度カメラ前提の v1 から
> **通常の RGB カメラ（UVC/Webカメラ または 任意の mp4）のみ**で動くシステムへ転換した。
> 旧 D405/M3（面-面深度較正）パスは「休止」であり削除していない（付録A参照）。

## 1. ゴール
RGB 単眼動画（頭部装着等）から、**両手・指（各21関節×2手）の world 座標軌道**を推定する。
ベースは **HaWoR**。SLAM・カメラ↔world 合成・メトリックスケール推定は HaWoR が内部で行う。

## 用語
- **M2'**：本書での取込段階。録画（UVC）または既存 mp4 の受け入れ + 内部パラメータ取得。
- **エクスポート段**：HaWoR 出力をアダプタ経由で最終 npz に変換する段階（旧 M3+ に相当）。
- **α**：world 軌道に掛けるグローバルスケール係数。**本版では 1.0 固定**（HaWoR 内蔵スケールを信頼）。
- **Metric3D**：単眼メトリック深度推定モデル。HaWoR が SLAM スケール決定に内部使用する。
- **MANO**：手の形状・姿勢のパラメトリックモデル（778頂点・21関節を出力）。
- **整列（alignment）**・**面-面**・**デプスレンダ**：旧 D405 パスの用語（付録A）。

## 2. 方針（厳守）
- HaWoR 本体は**改造しない**（vendored コードへの変更はビルド/torch互換の最小パッチのみ。§4 の表と patches/ に記録）。
- **実寸スケールは HaWoR 内蔵（Metric3D）を信頼し α=1.0 固定**。
  ただし出力 npz とエクスポート段は ×α の機構を温存する（将来、手実寸較正・マーカー較正・
  D405 復活のいずれでも α の差し替えだけで実寸化できる）。
- 内部パラメータは**チェスボードキャリブレーションで実測**し HaWoR に渡す
  （未指定だと HaWoR は焦点距離 600px にフォールバックし精度が落ちる）。
- 入力動画は intrinsics に基づき**理想ピンホール化（歪み補正＋主点センタリング）してから** HaWoR に渡す
  （HaWoR の内部仮定 fx=fy・主点=中央・歪み0 と入力を一致させる衛生措置。run_pipeline.py が自動実施、
  --no_rectify で無効）。**ただし実測では本補正によるオーバーレイ整合の改善は確認されていない**
  （主点26pxずれ環境で補正前後の投影差 1.6〜5.6px。系統誤差は HaWoR の並進回帰自体が支配的 — §7）。
- 既存実装を流用し、新規実装は最小限。
- **やらない**：HaWoR 内部改変／時間変化α・再積分／位置ドリフト補正／独自 SLAM／独自スケール推定。

## 3. 全体アーキテクチャ
```
UVCカメラ ─ scripts/record_rgb.py ─► rgb.mp4 ─┐
任意の mp4 ──────────────────────────────────┤
チェスボード動画 ─ scripts/calibrate_camera.py ─► intrinsics.json（fx,fy,cx,cy。任意だが強く推奨）
                                      ▼
        tools/hawor_infer.py（HaWoR 無改造: 検出→MANO→masked DROID-SLAM＋Metric3Dスケール→world合成）
                                      ▼
        handtraj.HaworSequence（§5 アダプタ）─► tools/overlay_check.py（関門③': 投影一致の目視確認）
                                      ▼
        scripts/export_trajectory.py: world_true = α · world_joints   （本版は α=1.0）
                                      ▼
        出力: world_trajectory.npz — joints[T,2,21,3](m), alpha, valid[T], valid_per_hand[T,2]
```
- スケールの出どころ：HaWoR は masked DROID-SLAM の軌道スケールを Metric3D の単眼メトリック深度で
  推定する（SLAM npz の `scale`）。よって world 出力は「近似メートル」。誤差は §7。

## 4. 環境（構築済み・記録）
RTX 5080 (sm_120) / Ubuntu 24.04 / Python 3.12 / torch 2.7.1+cu128。構築手順は README.md。
MANO は登録制のため `tools/prepare_mano.py` 経由で配置（chumpy 除去変換、直接配置禁止）。

### vendored コードへの最小パッチ（HaWoR アルゴリズムは無改造・実体は patches/）
| ファイル | 変更 | 理由 |
|---|---|---|
| `third_party/HaWoR/thirdparty/DROID-SLAM/setup.py` | `-gencode sm_60..86` ハードコード削除 | `TORCH_CUDA_ARCH_LIST=12.0` を有効化（無いと RTX 5080 で kernel image エラー） |
| 同 `src/correlation_kernels.cu` / `src/altcorr_kernel.cu` | `AT_DISPATCH...(x.type(),` → `x.scalar_type()` | torch 2.x で旧 API 廃止 |
| 同 `thirdparty/lietorch/.../lietorch_{gpu.cu,cpu.cpp}` | `DISPATCH...(group_id, x.type(),` → `x.scalar_type()` | 同上 |

- 再クローン時は `bash patches/apply_patches.sh` で再適用（適用済み検知付き）。
- torch≥2.6 の `torch.load` weights_only 問題は `tools/hawor_infer.py` 内のプロセス限定
  monkeypatch で対応（vendored 改変なし）。
- 重み取得の記録: HF 系（detector.pt/hawor.ckpt/infiller.pt/model_config.yaml）は wget、
  droid.pth / Metric3D は gdown（Google Drive ID は HaWoR README 記載のリンク）。
- 注意: mediapipe を pip で入れると numpy が 2.x に上がり環境が壊れる（numpy<2 必須）。
  診断等で使う場合は別プロセス分離か、使用後に numpy==1.26.4 / opencv-python==4.11.0.86 を再固定。

## 5. インターフェース
- **M2' 出力**：`<take>/rgb.mp4`（HaWoR 入力）／`<take>/intrinsics.json`（任意・推奨）＝
  `{width,height,fx,fy,cx,cy,model,coeffs,fps}`（旧版から depth_scale を除いた同形式）。
- **アダプタ `adapters/load_hawor_frames.py`**（実装済み・一本化・互換シム）：
  実体は `handtraj/hawor_adapter.py` の `HaworSequence` クラス（2026-06 リファクタで
  `handtraj/` パッケージにライブラリ化。API 契約は `handtraj/CONTRACT.md`。
  CLI は全て薄いラッパーとして互換維持）。フレーム表現の仕様は以下の通り不変:

  | フィールド | 用途 | 内容 |
  |---|---|---|
  | `idx:int` | 同期 | 動画フレーム番号（0始まり） |
  | `verts_cam_m:(778,3)` | 検証/将来の較正 | MANO頂点（カメラ座標・m・HaWoR尺度） |
  | `faces:(F,3)` | 検証/将来の較正 | MANO面（手首閉鎖面込み F=1552） |
  | `world_joints:(2,21,3)` | エクスポート | world手関節（HaWoR尺度、[left,right]） |
  | `hand` | 補助 | left / right |

  - カメラ規約は **OpenCV系（+Z前方・y下・u=fx·X/Z+cx）で実証済み**（v1 関門③）。
  - 補助 API: `load_world_joints(seq)` → `joints[T,2,21,3], valid[T,2]`／`load_hawor_outputs(seq)`。
- **最終出力 `world_trajectory.npz`**：`joints[T,2,21,3]`(m)・`alpha`(=1.0)・`valid[T]`・`valid_per_hand[T,2]`。
  invalid な手の関節は NaN。

## 6. 実装タスク（この順・完了済み）
1. `scripts/calibrate_camera.py`：チェスボード動画/画像 → `intrinsics.json`。`--selftest` 付き。
2. `scripts/record_rgb.py`：UVC 録画 → mp4（実効 fps 計測・警告付き。HaWoR は 30fps 前提）。
3. `scripts/export_trajectory.py`：`world_joints × α`（既定 α=1.0、`--alpha`/`--m3_report` で差し替え可）→ npz + 診断。
4. `scripts/run_pipeline.py`：mp4 取込 → HaWoR → overlay → エクスポート（intrinsics 無しでも動くが警告）。
5. 実カメラで一気通貫（キャリブ → 録画 → 軌道出力）。
6. 2D キーポイント並進リファイン（`handtraj/refine.py` + `tools/detect_keypoints_2d.py`[隔離venv] +
   `scripts/refine_trajectory.py` / `run_pipeline --refine`。npz 追加キー: refined / delta_t_world）。

### 関門（通るまで先へ進まない）
- **関門②'**：`python scripts/calibrate_camera.py --selftest` が通る（合成チェスボードで fx 誤差 <1%）。
- **関門③'**：実測 intrinsics を使った投影オーバーレイが手の輪郭に一致する。
- **最終**：任意動画の一気通貫で npz 生成。出力の手サイズ（手首〜中指MCP等の骨長）が解剖学的に妥当
  （目安: 手長 16〜20cm）であること。

## 7. 既知の限界
- **実寸精度は Metric3D 依存（誤差 5〜15% 想定）**。実寸が要件化したら α 較正（手実寸測定／
  既知サイズマーカー／D405 復活=付録A）を追加する。npz の `alpha` フィールドはそのための受け皿。
- **画像面での手位置の整合は素の状態で ~25〜55px（約1.2〜2.6cm @0.5m）**：MediaPipe を独立基準にした実測値
  （2026-06-05, DJI Action2）。内訳は HaWoR の並進回帰誤差・infiller の時間平滑化が支配的で、
  較正パラメータでは詰められない（主点±18px相当を変えても測定可能な改善なし）。
  → **対策実装済み: 2D キーポイント並進リファイン**（handtraj/refine.py, `--refine`）。
  ホールドアウト評価で 55→16px に改善。残る ~16px は MANO 関節中心と MediaPipe ランドマークの
  **定義差**（手首18px/MCP25px が支配、PIP/DIP は12px）であり、軌道誤差ではない。
  奥行き成分の補正は 2D 観測の情報限界により部分的（selftest 実測: 面内96%/奥行き65%復元）。
- 位置・回転ドリフトは未補正（HaWoR任せ）。長尺の絶対精度は静止区間で要実測。
- 手の検出失敗・画面外で valid=False（NaN）。30fps 以外の入力は ffmpeg 再サンプルで 1:1 対応が崩れ得る。

## 8. 検証実績（関門の状態・2026-06-05 時点）
- v1 関門①（sm_120 ビルド・import）: **PASS**
- v1 関門②（面-面 selftest）: **PASS**（α=0.85 復元・std 0.0001）
- v1 関門③（投影オーバーレイ）: **PASS** — カメラ規約 OpenCV 系で確定
- v2 関門②'（calibrate_camera --selftest）: **PASS** — fx 誤差 0.01〜0.39%（2ケース）
- v2 関門③'（実測 intrinsics での投影一致）: **PASS** — DJI Action2 実録画で確認
- v2 最終（一気通貫 + 手長妥当性）: **PASS** — example + 実録画（手長 17.4〜18.4cm）
- M3 統合検証（実メッシュ×合成深度）: α=0.85 を誤差 0.00% で復元
- リファクタ回帰（2026-06）: take03 npz 数値一致・overlay PNG バイト一致・契約準拠レビュー合格
- 関門④a（リファイン selftest）: **PASS** — 人工オフセット±2cm を面内95.7%復元、残差32.5→3.4px
- 関門④b（ホールドアウト・非循環評価）: **PASS** — 偶数フレーム学習→奇数フレームで 55.0→16.1px
  （fit 15.8px とほぼ同値=過学習なし）。残差フロアは関節定義差（§7）
- 関門④c（副作用なし）: **PASS** — 手長完全不変（剛体シフト）・ジャーク +3.5%（σ_vel=0.003 採用）

## 付録A: 休止中の D405 パス（v1・`legacy/` 配下）
- 構成：`legacy/record_d405.py`（.bag録画）→ `legacy/split_bag.py`（RGB+整列深度+intrinsics 分離）→
  `legacy/m3_surface_calibrate.py`（**面-面レンダリング**でグローバルα較正）→ ×α。
- v1 関門通過実績：①sm_120 ビルド ②面-面 selftest（α=0.85 復元）③投影一致
  ＋実メッシュ×合成深度の統合検証（α 誤差 0.00%）。
- 復活手順：D405 入手 → v1 通り録画・分離 → `m3_surface_calibrate.py` で α 算出 →
  `export_trajectory.py --m3_report <report>` に差し替えるだけ（インターフェースは温存済み）。
