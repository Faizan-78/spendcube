#!/usr/bin/env python3
"""Measure the tool against Company A's labels.

    python evaluate.py --config configs/A.yaml --data data/company_A --labels data/company_A/labels_A.csv

Runs the pipeline twice: once WITH labels hidden (tree-only classification, unguided de-duplication) and
reports, on the ERP records:
  * classification accuracy at L1/L2/L3, by count and by spend
  * supplier and item de-duplication: pairwise precision / recall / F1 and adjusted Rand index vs the golden keys
plus a 5-fold cross-validated accuracy for the label-trained classifier.
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from spendcube.ingest import load_config
from spendcube.pipeline import run


def pairwise(pred: pd.Series, gold: pd.Series):
    """Pairwise precision/recall over rows (pairs of rows put in the same group)."""
    df = pd.DataFrame({"p": pred.values, "g": gold.values})
    c2 = lambda n: n * (n - 1) / 2
    tp = c2(df.groupby(["p", "g"]).size()).sum()
    pp = c2(df.groupby("p").size()).sum()
    gp = c2(df.groupby("g").size()).sum()
    prec = tp / pp if pp else 1.0
    rec = tp / gp if gp else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, adjusted_rand_score(df["g"], df["p"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--prefix", default="ERP:")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(a.config)
    cfg["reference_dir"] = None
    lab = pd.read_csv(a.labels, dtype=str).set_index("record_id")
    lv = [c for c in lab.columns if c.startswith("category_l")]

    t = time.time()
    out = run(cfg, a.data, "/tmp/eval_nolabels.csv", use_labels=False, audit=False).set_index("record_id")
    secs = time.time() - t
    erp = out[out.index.str.startswith(a.prefix) & (out["include"] == 1)]
    g = lab.loc[erp.index]
    w = erp["spend"].abs()
    print(f"\n== A with labels hidden ({secs:.0f}s for {len(out)} records) ==")
    print(f"included ERP lines scored: {len(erp)}")
    for k, c in enumerate(lv):
        ok = (erp[f"l{k + 1}"].values == g[c].values)
        print(f"  L{k + 1} accuracy: {ok.mean():.3f} by count, {(ok * w).sum() / w.sum():.3f} by spend")
    for col, gcol in (("supplier_key", "supplier_key"), ("item_key", "item_key")):
        m = g[gcol].notna()
        p, r, f, ari = pairwise(erp.loc[m, col], g.loc[m, gcol])
        print(f"  {col}: precision {p:.3f} recall {r:.3f} F1 {f:.3f} ARI {ari:.3f} "
              f"(groups: ours {erp.loc[m, col].nunique()}, golden {g.loc[m, gcol].nunique()})")

    # label-trained classifier, 5-fold CV on ERP lines
    from spendcube.classify import Tree, _features, _fit_predict
    import pickle
    tree = Tree(Path(a.data) / cfg["taxonomy"]["file"], cfg["taxonomy"]["levels"])
    idx = tree.index()
    y = np.array([idx[tuple(r)] for r in lab[lv].itertuples(index=False, name=None)])
    raw = pd.read_csv(Path(a.data) / cfg["sources"][0]["file"], dtype=str).fillna("")
    f = cfg["sources"][0]["fields"]
    docs = pd.DataFrame({"text": raw[f["description"]].agg(" ".join, axis=1), "ctx": raw[cfg["sources"][0]["classify"]["text"]].agg(" ".join, axis=1),
                         "cats": ("sup=" + raw[f["supplier_name"]].str.replace(r"\s+", "_", regex=True) + " item=" + raw[f["item_code"]])})
    X = _features(docs)
    fold = np.random.RandomState(0).randint(0, 5, len(y))
    acc = []
    for k in range(5):
        tr = np.where(fold != k)[0]
        P = _fit_predict(X, tr, y[tr], len(tree))
        acc.append((P[fold == k].argmax(1) == y[fold == k]).mean())
    print(f"\n== label-trained classifier, 5-fold CV on ERP lines: L3 accuracy {np.mean(acc):.3f} ==")


if __name__ == "__main__":
    main()
