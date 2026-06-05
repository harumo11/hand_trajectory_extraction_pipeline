#!/usr/bin/env python3
"""
m3_surface_calibrate.py — 面-面レンダリングによるスケール較正（面-面方式・最終版）

考え方・原理は §3 を参照（表面 vs 表面で δ を消し、原点まわりスケール α を回帰）。

依存: numpy, opencv-python, matplotlib （torch/CUDA非依存。レンダは純numpyの自前ラスタライザ）

使い方:
  # 自己検証（HaWoR不要）: 楕円柱を「ひねり＋距離変動」させ、面-面ならαが一定復元されることを実証
  python m3_surface_calibrate.py --selftest

  # 実データ:
  python m3_surface_calibrate.py --hawor <out> --depth_dir <take>/depth \
                                 --intrinsics <take>/intrinsics.json --out <take>/m3s

前提(要確認): HaWoR のカメラ規約は OpenCV系（+Z 前方, y 下向き, 画素 u=fx*X/Z+cx）とみなす。
            メッシュ頂点はカメラ座標・メートルで与える。実出力で要検証。
"""
import argparse, json, os, sys
import numpy as np

D_MIN_M, D_MAX_M = 0.07, 0.50   # D405有効レンジ


# ============ 純numpy デプスラスタライザ ============
def _edge(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def render_depth(V, F, K, H, W):
    """三角メッシュ(カメラ座標, m)を透視投影しデプスバッファ(Z, m)を返す。空はinf。"""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Zc = V[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * V[:, 0] / Zc + cx
        v = fy * V[:, 1] / Zc + cy
    Z = np.full((H, W), np.inf, np.float32)
    for tri in F:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        if Zc[i] <= 1e-6 or Zc[j] <= 1e-6 or Zc[k] <= 1e-6:
            continue  # 背面/カメラ後方は除外
        x0, y0, z0 = u[i], v[i], Zc[i]
        x1, y1, z1 = u[j], v[j], Zc[j]
        x2, y2, z2 = u[k], v[k], Zc[k]
        minx = int(max(0, np.floor(min(x0, x1, x2))))
        maxx = int(min(W - 1, np.ceil(max(x0, x1, x2))))
        miny = int(max(0, np.floor(min(y0, y1, y2))))
        maxy = int(min(H - 1, np.ceil(max(y0, y1, y2))))
        if minx > maxx or miny > maxy:
            continue
        area = _edge(x0, y0, x1, y1, x2, y2)
        if abs(area) < 1e-9:
            continue
        xs = np.arange(minx, maxx + 1) + 0.5
        ys = np.arange(miny, maxy + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        w0 = _edge(x1, y1, x2, y2, gx, gy)
        w1 = _edge(x2, y2, x0, y0, gx, gy)
        w2 = _edge(x0, y0, x1, y1, gx, gy)
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) if area > 0 \
            else (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
        if not inside.any():
            continue
        b0, b1, b2 = w0 / area, w1 / area, w2 / area
        invz = b0 / z0 + b1 / z1 + b2 / z2          # 透視補正(1/Zは画面で線形)
        zt = 1.0 / invz
        sub = Z[miny:maxy + 1, minx:maxx + 1]
        upd = inside & (zt < sub)
        sub[upd] = zt[upd]
        Z[miny:maxy + 1, minx:maxx + 1] = sub
    return Z


# ============ 頑健スケール推定（原点通過: 切片なし） ============
def robust_scale(z_model, z_rs):
    r = z_rs / z_model
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)) * 1.4826)
    keep = np.abs(r - med) <= (3 * mad + 1e-9)        # 外れ値除去
    alpha = float(np.median(r[keep]))
    return alpha, {"n": int(r.size), "n_kept": int(keep.sum()),
                   "alpha": alpha, "ratio_mad": mad}


def read_depth_m(png_path, depth_scale):
    import cv2
    raw = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    d = raw.astype(np.float32) * depth_scale
    d[raw == 0] = np.nan
    return d


def collect_pairs(z_model, z_rs):
    """重なり・有効・レンジ内の画素ペア(Z_model, Z_rs)を返す。"""
    m = np.isfinite(z_model) & np.isfinite(z_rs)
    m &= (z_rs >= D_MIN_M) & (z_rs <= D_MAX_M)
    m &= (z_model > 1e-6)
    return z_model[m], z_rs[m]


# ============ HaWoR アダプタ ============
# 仕様は §5 を正本とする。実体は adapters/load_hawor_frames.py に一本化済み（ここでは委譲のみ）。
def load_hawor_frames(hawor_output_path):
    """仕様は §5 を参照。実体は adapters/load_hawor_frames.py（一本化モジュール）。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from adapters.load_hawor_frames import load_hawor_frames as _impl
    return _impl(hawor_output_path)


def calibrate(frames, depth_dir, K, depth_scale, H, W, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    all_m, all_r, per_frame = [], [], []
    for fr in frames:
        zmod = render_depth(np.asarray(fr["verts_cam_m"]), np.asarray(fr["faces"]), K, H, W)
        zrs = read_depth_m(os.path.join(depth_dir, f"depth_{fr['idx']:06d}.png"), depth_scale)
        m, r = collect_pairs(zmod, zrs)
        if m.size >= 50:
            a_f, _ = robust_scale(m, r)
            per_frame.append({"idx": fr["idx"], "alpha": a_f, "n": int(m.size)})
            all_m.append(m); all_r.append(r)
    if not all_m:
        return {"status": "no_valid_pixels"}
    M, R = np.concatenate(all_m), np.concatenate(all_r)
    alpha, stats = robust_scale(M, R)
    report = {"status": "ok", "alpha_global": alpha, **stats,
              "alpha_per_frame": per_frame}
    with open(os.path.join(out_dir, "m3_surface_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "alpha_per_frame"},
                     indent=2, ensure_ascii=False))
    return report


# ============ selftest: ひねり耐性の実証 ============
def make_elliptic_tube(a, b, length, n_theta=40, n_len=4):
    ys = np.linspace(-length / 2, length / 2, n_len)
    phis = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    V = np.array([[a * np.cos(p), y, b * np.sin(p)] for y in ys for p in phis])
    F = []
    for r in range(n_len - 1):
        for c in range(n_theta):
            c2 = (c + 1) % n_theta
            p00, p01 = r * n_theta + c, r * n_theta + c2
            p10, p11 = (r + 1) * n_theta + c, (r + 1) * n_theta + c2
            F += [[p00, p10, p11], [p00, p11, p01]]
    return V, np.array(F)


def roty(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def run_selftest():
    H, W = 240, 320
    fx = fy = 300.0
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]])
    alpha_true = 0.85
    a, b = 0.030, 0.018      # 楕円半径: 幅3cm/厚み1.8cm（真円でない＝ひねりでδ変化）
    V0, F = make_elliptic_tube(a, b, length=0.06)

    n = 36
    thetas = np.linspace(0, np.pi, n)              # ひねり 0..180°
    dists = 0.12 + 0.20 * (0.5 + 0.5 * np.sin(np.linspace(0, 7, n)))  # 距離も変動

    surf_alphas, naive_alphas = [], []
    M_all, R_all = [], []
    for th, D in zip(thetas, dists):
        Vt = (roty(th) @ V0.T).T + np.array([0, 0, D])    # 真メッシュ(ひねり＋距離)
        Vm = Vt / alpha_true                               # モデル＝真を1/αで縮尺(誤スケール)
        z_true = render_depth(Vt, F, K, H, W)
        # センサ実測(表面)にノイズ
        z_rs = z_true + np.where(np.isfinite(z_true),
                                 np.random.normal(0, 0.001, z_true.shape), 0)
        z_model = render_depth(Vm, F, K, H, W)
        m, r = collect_pairs(z_model, z_rs)
        if m.size >= 50:
            af, _ = robust_scale(m, r)
            surf_alphas.append(af); M_all.append(m); R_all.append(r)
        # 比較用: 素朴な「中心 vs 表面」比（ひねりでバイアスするはず）
        delta = np.sqrt((a * np.sin(th))**2 + (b * np.cos(th))**2)  # 表面-中心オフセット
        rs_surface_center = D - delta
        model_center = D / alpha_true
        naive_alphas.append(rs_surface_center / model_center)

    M, R = np.concatenate(M_all), np.concatenate(R_all)
    alpha_est, _ = robust_scale(M, R)
    print(f"[selftest] true alpha          = {alpha_true:.3f}")
    print(f"[selftest] 面-面 推定 alpha     = {alpha_est:.3f}  (ひねり0-180°・距離変動でも一定)")
    print(f"[selftest] 面-面 フレーム毎 std = {np.std(surf_alphas):.4f}  (≒0が理想)")
    print(f"[selftest] 素朴(中心)比 範囲    = {min(naive_alphas):.3f}〜{max(naive_alphas):.3f}"
          f"  (ひねりで{(max(naive_alphas)-min(naive_alphas))/alpha_true*100:.0f}%変動＝バイアス)")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(np.degrees(thetas), surf_alphas, "o-", label="面-面 (頑健)")
        plt.plot(np.degrees(thetas), naive_alphas, "x--", label="素朴 中心vs表面 (ひねりで偏る)")
        plt.axhline(alpha_true, color="k", ls=":", label="true")
        plt.xlabel("手首ひねり角 [deg]"); plt.ylabel("推定 alpha"); plt.legend()
        plt.title("面-面はひねりに不変 / 素朴比はδ変動で偏る")
        plt.tight_layout(); plt.savefig("m3_selftest_twist.png", dpi=120)
        print("[selftest] 図: m3_selftest_twist.png")
    except Exception as e:
        print(f"[WARN] plot skip: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--hawor"); ap.add_argument("--depth_dir")
    ap.add_argument("--intrinsics"); ap.add_argument("--out", default="m3s_out")
    args = ap.parse_args()
    if args.selftest:
        return run_selftest()
    assert args.hawor and args.depth_dir and args.intrinsics
    with open(args.intrinsics) as f:
        mt = json.load(f)
    K = np.array([[mt["fx"], 0, mt["cx"]], [0, mt["fy"], mt["cy"]], [0, 0, 1.0]])
    H, W = int(mt["height"]), int(mt["width"])
    frames = load_hawor_frames(args.hawor)
    calibrate(frames, args.depth_dir, K, float(mt["depth_scale_m_per_unit"]), H, W, args.out)


if __name__ == "__main__":
    sys.exit(main())
