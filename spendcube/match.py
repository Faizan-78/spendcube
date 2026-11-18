"""Count each purchase once.

Three kinds of repeats are detected, all config-driven:
  * the same invoice line exported more than once inside one source (REPEAT_EXPORT);
  * byte-identical rows inside one source (EXACT_ROW);
  * a record of the secondary system that is already present in the system of record (CROSS_SYSTEM).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from .textnorm import text_key

log = logging.getLogger("spendcube")


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.5
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _cents(x) -> int:
    return int(round(float(x) * 100))


class SequenceHint:
    """Record ids in both systems grow with time. Learn 'secondary order -> primary order' from
    confirmed matches and use it to reject coincidental same-supplier-same-amount pairs."""

    def __init__(self, pairs, min_pairs=50):
        self.ok = False
        if len(pairs) < min_pairs:
            return
        a = np.array(sorted(pairs), dtype=float)
        if np.isnan(a).any() or abs(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) < 0.9:
            return
        y = pd.Series(a[:, 1]).rolling(201, center=True, min_periods=1).median().to_numpy()
        self.x, self.y = a[:, 0], y
        resid = np.abs(a[:, 1] - y)
        self.tol = float(np.quantile(resid, 0.995) * 1.25 + 1)
        self.ok = True

    def distance(self, x, y) -> float:
        if not self.ok or pd.isna(x) or pd.isna(y):
            return 0.0
        return abs(float(np.interp(x, self.x, self.y)) - y)

    def accepts(self, x, y) -> bool:
        return (not self.ok) or self.distance(x, y) <= self.tol


def find_duplicates(df: pd.DataFrame, cfg: dict):
    """Returns (dup_of, method, links). dup_of holds the rid of the record that is counted instead."""
    dup_of = pd.Series("", index=df.index, dtype="object")
    method = pd.Series("", index=df.index, dtype="object")
    links = []
    run = df[~df["ref"]]
    dcfg = cfg.get("duplicates") or {}
    approved = (run["status_class"] == "INCLUDE")

    # 1. byte-identical rows
    if dcfg.get("exact_rows", False):
        for _, idx in run.groupby(["source", "row_hash"], sort=False).groups.items():
            if len(idx) > 1:
                idx = sorted(idx, key=lambda i: (not approved[i], i))
                dup_of.loc[idx[1:]] = df.at[idx[0], "rid"]
                method.loc[idx[1:]] = "EXACT_ROW"
        log.info("duplicates: %d byte-identical rows", int((method == "EXACT_ROW").sum()))

    # 2. the same invoice line exported twice inside one source
    group_po = df["po"].copy()
    group_of = pd.Series(df.index.to_numpy().copy(), index=df.index)  # copy: never alias the index buffer
    for source, keys in (dcfg.get("repeated_export") or {}).items():
        sub = run[(run["source"] == source) & (dup_of[run.index] == "")]
        cols = [{"supplier": "supplier_key", "invoice_number": "inv", "net_amount": "net_r", "quantity": "qty_r",
                 "description": "desc_key", "po": "po"}.get(k, k) for k in keys]
        sub = sub.assign(net_r=sub["net"].round(2), qty_r=sub["qty"].round(4).fillna(-1e18))
        if "desc_key" in cols:
            sub["desc_key"] = sub["desc"].map(text_key)
        sub = sub[(sub["inv"] != "") & sub["net"].notna()] if "inv" in cols else sub
        n = 0
        for _, idx in sub.groupby(cols, sort=False).groups.items():
            if len(idx) < 2:
                continue
            idx = sorted(idx, key=lambda i: (not approved[i], df.at[i, "order"] if pd.notna(df.at[i, "order"]) else 0, i))
            pos = [p for p in df.loc[idx, "po"] if p]
            if pos:
                group_po.loc[idx] = max(set(pos), key=pos.count)
            group_of.loc[idx] = idx[0]
            dup_of.loc[idx[1:]] = df.at[idx[0], "rid"]
            method.loc[idx[1:]] = "REPEAT_EXPORT"
            links += [(idx[0], j) for j in idx[1:]]
            n += len(idx) - 1
        log.info("duplicates: source %s has %d repeated exports of the same invoice line", source, n)

    # 3. cross-system
    m = cfg.get("match")
    if m:
        prim = run[(run["source"] == m["count"]) & run["net"].notna()]
        sec = run[(run["source"] == m["against"]) & run["net"].notna() & (group_of[run.index] == run.index)
                  & (method[run.index] != "EXACT_ROW")]
        tol = _cents(m.get("amount_tolerance", 0.02))
        by_po, by_sup = defaultdict(list), defaultdict(lambda: defaultdict(list))
        p_net = {i: _cents(v) for i, v in prim["net"].items()}
        for i, po, sk in zip(prim.index, prim["po"], prim["supplier_key"]):
            if po:
                by_po[po].append(i)
            by_sup[sk][p_net[i]].append(i)
        tkey = {}

        def tk(i):
            if i not in tkey:
                tkey[i] = text_key(df.at[i, "desc"])
            return tkey[i]

        claimed, matched, how = set(), {}, {}
        alt_rates = [0.0, float(m.get("default_tax_rate", 0.0))]

        def best(u, cands, nets, need_text=False):
            top, top_s = None, -1.0
            for c in cands:
                if c in claimed or min(abs(p_net[c] - n) for n in nets) > tol:
                    continue
                qa, qb = df.at[u, "qty"], df.at[c, "qty"]
                q_ok = pd.notna(qa) and pd.notna(qb) and abs(qa - qb) < 1e-6
                ts = _sim(tk(u), tk(c))
                if need_text and not (q_ok and ts >= 0.5) and ts < 0.8:
                    continue
                s = ts + (1.0 if q_ok else 0.0)
                if df.at[u, "po_line"] and df.at[u, "po_line"] == df.at[c, "po_line"]:
                    s += 2.0
                if s > top_s:
                    top, top_s = c, s
            return top

        order = sorted(sec.index, key=lambda i: (not approved[i], i))
        # pass 1: PO reference present and known to the system of record
        for u in order:
            po = group_po[u]
            if not po or po not in by_po:
                continue
            c = best(u, by_po[po], [_cents(df.at[u, "net"])])
            tag = "PO+AMOUNT"
            if c is None and pd.notna(df.at[u, "gross"]):
                nets = [_cents(df.at[u, "gross"] / (1 + r)) for r in alt_rates]
                c, tag = best(u, by_po[po], nets, need_text=True), "PO+AMOUNT(OTHER_TAX_RATE)"
            if c is not None:
                claimed.add(c); matched[u] = c; how[u] = tag
        hint = SequenceHint([(df.at[u, "order"], df.at[c, "order"]) for u, c in matched.items()]) \
            if m.get("sequence_hint", True) else SequenceHint([])
        if hint.ok:
            log.info("matching: id-sequence hint learned from %d PO matches (tolerance %.0f)", len(matched), hint.tol)
        # pass 2: no PO reference -> supplier + amount + quantity/text (+ sequence proximity)
        for u in order:
            if u in matched or group_po[u]:
                continue
            n = _cents(df.at[u, "net"])
            cands = [c for d in range(-tol, tol + 1) for c in by_sup[df.at[u, "supplier_key"]].get(n + d, [])]
            cands = [c for c in cands if hint.accepts(df.at[u, "order"], df.at[c, "order"])]
            if hint.ok:
                cands.sort(key=lambda c: hint.distance(df.at[u, "order"], df.at[c, "order"]))
                keep = None
                for c in cands:  # nearest in sequence that also agrees on quantity/text
                    if best(u, [c], [n], need_text=True) is not None:
                        keep = c
                        break
                c = keep
            else:
                c = best(u, cands, [n], need_text=True)
            if c is not None:
                claimed.add(c); matched[u] = c; how[u] = "SUPPLIER+AMOUNT"
        members_of = defaultdict(list)
        for i, g in group_of[group_of != group_of.index].items():
            members_of[g].append(i)
        for u, c in matched.items():
            members = [u] + members_of.get(u, [])
            dup_of.loc[members] = df.at[c, "rid"]
            method.loc[members] = how[u]
            links.append((c, u))
        log.info("matching: %d of %d distinct %s invoice lines are already in %s (%s)", len(matched), len(sec),
                 m["against"], m["count"], dict(pd.Series(how).value_counts()) if how else {})
    return dup_of, method, links
