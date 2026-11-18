"""Decide what counts as spend. Every exclusion carries one of the four reason codes."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import yaml

from .textnorm import text_key

log = logging.getLogger("spendcube")
RES = Path(__file__).parent / "resources"


def status_class(status_raw: pd.Series, status_map: dict) -> pd.Series:
    """INCLUDE / CANCELLED / NOT_APPROVED per record. Blank status = the source has no status = INCLUDE."""
    uniq = {}
    for raw in status_raw.fillna("").unique():
        s = str(raw).strip().lower()
        cls = "INCLUDE" if s == "" else None
        for code in ("CANCELLED", "NOT_APPROVED", "INCLUDE"):
            if cls is None and any(k in s for k in status_map.get(code, [])):
                cls = code
        if cls is None:
            cls = "UNKNOWN"
        uniq[raw] = cls
    unknown = sorted(k for k, v in uniq.items() if v == "UNKNOWN")
    if unknown:
        log.warning("status values not recognised (treated as NOT_APPROVED; map them under status_map: in the config): %s", unknown)
    return status_raw.fillna("").map(uniq)


def _load_rules(cfg: dict, tree_words: set) -> list[dict]:
    np_cfg = cfg.get("non_procurement") or {}
    rules = []
    if np_cfg.get("use_builtin", True):
        for r in yaml.safe_load((RES / "non_procurement.yaml").read_text(encoding="utf-8"))["rules"]:
            if r["name"] in (np_cfg.get("disable") or []):
                continue
            hit = [g for g in r.get("guard", []) if any(g in w for w in tree_words)]
            if hit:
                log.info("non-procurement rule '%s' switched off: the category tree has a category about %s", r["name"], hit)
                continue
            rules.append(r)
    for i, r in enumerate(np_cfg.get("rules") or []):
        rules.append({"name": r.get("name", f"config_rule_{i + 1}"), **r})
    names = [n for n in (np_cfg.get("company_names") or []) if n]
    if names:
        rules.append({"name": "own_company_as_supplier", "fields": ["supplier_name"],
                      "regex": "|".join(re.escape(n) for n in names)})
    return rules


def non_procurement(df: pd.DataFrame, cfg: dict, tree_words: set) -> pd.Series:
    """Name of the rule that marks the record as NON_PROCUREMENT ('' if none)."""
    hit = pd.Series("", index=df.index, dtype="object")
    for r in _load_rules(cfg, tree_words):
        rx = re.compile(r["regex"], re.I)
        un = re.compile(r["unless"], re.I) if r.get("unless") else None
        for field in r.get("fields", ["desc"]):
            if field not in df.columns:
                continue
            vals = df[field].fillna("").astype(str)
            uniq = {v: bool(rx.search(v)) and not (un and un.search(v)) for v in vals.unique() if v}
            m = vals.map(uniq).fillna(False).astype(bool) & (hit == "")
            hit[m] = r["name"]
    return hit


def learned_exclusions(df: pd.DataFrame, labelled_include: pd.Series, min_support=3) -> pd.Series:
    """Generalise analyst labels: an account description / line description that the analyst excluded
    every time (at least min_support times) is excluded everywhere."""
    hit = pd.Series("", index=df.index, dtype="object")
    lab = labelled_include.dropna()
    if lab.empty:
        return hit
    for field, norm in (("gl_desc", lambda s: s.str.strip().str.lower()), ("desc", lambda s: s.map(text_key))):
        key = norm(df[field].fillna("").astype(str))
        g = pd.DataFrame({"k": key.loc[lab.index], "inc": lab.astype(float)})
        g = g[g["k"].str.len() >= 3].groupby("k")["inc"].agg(["size", "sum"])
        bad = set(g[(g["size"] >= min_support) & (g["sum"] == 0)].index)
        m = key.isin(bad) & (hit == "")
        hit[m] = f"learned_from_labels:{field}"
    return hit
