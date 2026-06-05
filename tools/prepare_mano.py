#!/usr/bin/env python3
"""
prepare_mano.py — MANO 公式 pkl を chumpy 非依存に変換して HaWoR の所定位置へ配置する。

背景:
  MANO 公式配布の pkl は chumpy オブジェクトを含み、unpickle に chumpy が必要。
  chumpy は Python 3.11+ 非互換（inspect.getargspec / numpy 旧エイリアス）のため、
  chumpy の最小スタブで unpickle し、全フィールドを純 numpy に変換して保存し直す。
  （値は無変換。smplx==0.1.28 はこの形式をそのまま読める）

使い方:
  # MANO 公式サイト (https://mano.is.tue.mpg.de) から mano_v1_2.zip を取得後:
  python tools/prepare_mano.py --mano_zip ~/Downloads/mano_v1_2.zip
  # もしくは展開済みディレクトリ（models/MANO_*.pkl を含む）:
  python tools/prepare_mano.py --mano_dir ~/Downloads/mano_v1_2

配置先（HaWoR README 準拠）:
  third_party/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl
  third_party/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl
"""
import argparse
import os
import pickle
import sys
import tempfile
import types
import zipfile

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HAWOR_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "third_party", "HaWoR"))
DEST = {
    "MANO_RIGHT.pkl": os.path.join(HAWOR_ROOT, "_DATA", "data", "mano", "MANO_RIGHT.pkl"),
    "MANO_LEFT.pkl": os.path.join(HAWOR_ROOT, "_DATA", "data_left", "mano_left", "MANO_LEFT.pkl"),
}


def _install_chumpy_stub():
    """unpickle 専用の chumpy 最小スタブを sys.modules に登録する。"""
    class Ch:
        # pickle は通常 __init__ を呼ばず状態を復元するだけ。reconstructor 経由で
        # 引数付きで呼ばれるケースに備え、何でも受けて捨てる。
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.__dict__["_state"] = state

    def _module_getattr(name, _C=Ch):
        # __file__ / __path__ 等の特殊属性にクラスを返すと inspect が壊れるため、
        # 非公開名は正しく AttributeError にする。
        if name.startswith("_"):
            raise AttributeError(name)
        return _C

    for name in ("chumpy", "chumpy.ch", "chumpy.ch_ops", "chumpy.reordering"):
        mod = types.ModuleType(name)
        mod.__file__ = "<chumpy-stub>"
        mod.Ch = Ch
        mod.__getattr__ = _module_getattr
        sys.modules[name] = mod
    return Ch


def _to_numpy(v, Ch):
    if not isinstance(v, Ch):
        return v
    d = getattr(v, "__dict__", {})
    # 単純な chumpy.Ch: 実データは 'x'
    if isinstance(d.get("x"), np.ndarray):
        return np.asarray(d["x"])
    # 並べ替え系 (chumpy.reordering): 値 = a.ravel()[idxs].reshape(preferred_shape)
    # （MANO の shapedirs は 20 主成分から先頭 10 を選択するこの形式）
    if "idxs" in d and "a" in d:
        src = _to_numpy(d["a"], Ch)
        out = np.asarray(src).ravel()[np.asarray(d["idxs"])]
        ps = d.get("preferred_shape")
        return out.reshape(ps) if ps else out
    raise ValueError(f"chumpy オブジェクトから ndarray を抽出できません: keys={list(d)}")


def strip_chumpy(src_pkl, dst_pkl):
    Ch = _install_chumpy_stub()
    with open(src_pkl, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    out = {}
    for k, v in data.items():
        out[k] = _to_numpy(v, Ch)
    os.makedirs(os.path.dirname(dst_pkl), exist_ok=True)
    with open(dst_pkl, "wb") as f:
        pickle.dump(out, f, protocol=2)
    kinds = {k: type(v).__name__ for k, v in out.items()}
    print(f"[INFO] {os.path.basename(src_pkl)} -> {dst_pkl}")
    print(f"       fields: {kinds}")


def find_pkls(root):
    found = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn in DEST and fn not in found:
                found[fn] = os.path.join(dirpath, fn)
    return found


def verify_with_smplx():
    """変換後の pkl が smplx で読めるか確認（chumpy 不在の py3.12 で通れば成功）。"""
    import torch
    import smplx
    for hand, model_path in (("right", os.path.dirname(DEST["MANO_RIGHT.pkl"])),
                             ("left", os.path.dirname(DEST["MANO_LEFT.pkl"]))):
        layer = smplx.MANOLayer(model_path=model_path, is_rhand=(hand == "right"),
                                use_pca=False)
        out = layer(global_orient=torch.zeros(1, 1, 3, 3) + torch.eye(3),
                    hand_pose=(torch.zeros(1, 15, 3, 3) + torch.eye(3)),
                    betas=torch.zeros(1, 10),
                    transl=torch.zeros(1, 3), pose2rot=False)
        v = out.vertices.detach().numpy()
        assert v.shape == (1, 778, 3), f"頂点 shape 異常: {v.shape}"
        print(f"[INFO] smplx 読込検証 OK ({hand}): vertices {v.shape}, "
              f"手の幅 ~{(v[0,:,0].max()-v[0,:,0].min())*100:.1f} cm")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mano_zip", help="mano_v1_2.zip のパス")
    g.add_argument("--mano_dir", help="展開済み MANO ディレクトリ")
    ap.add_argument("--skip_verify", action="store_true")
    args = ap.parse_args()

    if args.mano_zip:
        tmp = tempfile.mkdtemp(prefix="mano_")
        with zipfile.ZipFile(args.mano_zip) as z:
            z.extractall(tmp)
        root = tmp
    else:
        root = args.mano_dir

    found = find_pkls(root)
    missing = [k for k in DEST if k not in found]
    if missing:
        print(f"[ERROR] {missing} が {root} 以下に見つかりません")
        return 1
    for fn, src in found.items():
        strip_chumpy(src, DEST[fn])
    if not args.skip_verify:
        verify_with_smplx()
    print("[INFO] MANO 配置完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
