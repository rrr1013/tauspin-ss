"""Train the kinematic (K) XGBoost classifier exactly as classify.py does for set K
(signal ggF HH vs all backgrounds, split 1, seed 0), save it, and write its score
for the given datasets (e.g. the spin-flat HH sample)."""
import argparse

import numpy as np
import xgboost as xgb

from classify import train_score


def kin_matrix(d, cols):
    return np.stack([d[c] for c in cols], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--apply", nargs="+", default=[])
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    cols = sorted(k for k in d.files if k.startswith("kin_"))
    y = (d["proc"] == "hh").astype(int)
    split = d["uid"] % 3
    import classify
    bst_holder = {}
    orig_train = xgb.train

    def capture(*a, **k):
        b = orig_train(*a, **k)
        bst_holder["b"] = b
        return b
    xgb.train = capture
    score, it = train_score(kin_matrix(d, cols), y, d["w"].astype(float), split == 1, split == 2, seed=0)
    xgb.train = orig_train
    b = bst_holder["b"]
    b.save_model(args.model)
    np.savez_compressed(args.model.replace(".json", "_scores.npz"), uid=d["uid"], proc=d["proc"], K=score,
                        best_iteration=it, columns=np.array(cols))
    for path in args.apply:
        a = np.load(path, allow_pickle=True)
        s = b.predict(xgb.DMatrix(kin_matrix(a, cols)), iteration_range=(0, it + 1))
        np.savez_compressed(path.replace(".npz", "_K.npz"), uid=a["uid"], proc=a["proc"], K=s)
        print(path, "scored", len(s))


if __name__ == "__main__":
    main()
