"""End-to-end run: ingest -> clean -> supplier keys -> count once -> item keys -> exclusions -> classify -> output."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dedupe, exclusions, match
from .classify import Tree, classify
from .ingest import ingest, load_status_map, read_csv
from .parsing import parse_number

log = logging.getLogger("spendcube")
OUT_COLS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend", "l1", "l2", "l3", "l4", "confidence"]


def _find(name, *dirs, columns=None) -> Path:
    """Look for `name` in each folder; if it is absent everywhere and `columns` is given, accept a renamed
    file in the run folder whose header holds those columns."""
    for d in dirs:
        if d and (Path(d) / name).exists():
            return Path(d) / name
    if columns and dirs and dirs[0]:
        from .ingest import find_by_columns
        alt = find_by_columns(Path(dirs[0]), columns)
        if alt is not None:
            log.warning("%s not found; using %s (its header matches)", name, alt.name)
            return alt
    raise FileNotFoundError(f"{name} not found in {[str(d) for d in dirs if d]}")


def _fx_table(cfg, dirs):
    fx = cfg.get("fx") or {}
    if not fx.get("file"):
        return {}, {}
    t = read_csv(_find(fx["file"], *dirs, columns=[fx.get("currency_column", "currency"), fx.get("month_column", "month"),
                                                    fx.get("rate_column", "rate")]))
    cur = t[fx.get("currency_column", "currency")].str.strip().str.upper()
    month = t[fx.get("month_column", "month")].str.strip().str[:7]
    rate = parse_number(t[fx.get("rate_column", "rate")])
    table = {(c, m): r for c, m, r in zip(cur, month, rate) if pd.notna(r) and r > 0}
    by_cur = {}
    for (c, m), r in table.items():
        by_cur.setdefault(c, []).append((pd.Period(m, "M").ordinal, r))
    return table, {c: sorted(v) for c, v in by_cur.items()}


def clean_money(df: pd.DataFrame, cfg: dict, dirs) -> pd.DataFrame:
    """Net of tax, in the reporting currency. Adds net, gross, fx_used and flags."""
    rep = cfg["reporting_currency"].upper()
    fxc = cfg.get("fx") or {}
    gross_basis = df["_amount_basis"] == "gross"
    net_cur = np.where(gross_basis, df["amount_raw"] / (1.0 + df["tax_rate"].fillna(0.0)), df["amount_raw"])
    table, by_cur = _fx_table(cfg, dirs)
    sanity = float(fxc.get("rate_sanity_factor", 3.0))
    multiply = fxc.get("quote", "units_per_reporting") == "reporting_per_unit"
    bad_date = df["flags"].str.contains("DATE_", na=False) | df["date"].isna()
    fx_date = df["date"].where(~bad_date, df["alt_date"]).fillna(df["date"])
    month = fx_date.dt.strftime("%Y-%m").fillna("")
    cur = df["currency"].fillna("").astype(str)
    same = (df["_amount_currency"] == "reporting") | (cur == "") | (cur == rep)
    used = np.ones(len(df)); flag = np.full(len(df), "", dtype=object)
    for i in np.where(~same.to_numpy())[0]:
        c, m, given = cur.iat[i], month.iat[i], df["fx_rate"].iat[i]
        tab, how = table.get((c, m)), ""
        if tab is None and c in by_cur and m:
            o = pd.Period(m, "M").ordinal
            tab = min(by_cur[c], key=lambda x: abs(x[0] - o))[1]
            how = "FX_NEAREST_MONTH"
        elif tab is None and c in by_cur:
            tab, how = by_cur[c][-1][1], "FX_NO_DATE"
        if pd.notna(given) and given > 0:
            if tab is not None and not (1 / sanity <= given / tab <= sanity):
                used[i], flag[i] = tab, "FX_RATE_IMPLAUSIBLE(table used)"
            else:
                used[i] = given
        elif tab is not None:
            used[i], flag[i] = tab, how
        else:
            used[i], flag[i] = np.nan, "FX_MISSING"
    net = net_cur * used if multiply else net_cur / used
    df = df.copy()
    df["net"] = np.where(same, net_cur, net)
    df["gross"] = np.where(gross_basis, np.where(same, df["amount_raw"], df["amount_raw"] * used if multiply else df["amount_raw"] / used), np.nan)
    df["fx_used"] = np.where(same, 1.0, used)
    fl = pd.Series(flag, index=df.index)
    df["flags"] = np.where(fl != "", np.where(df["flags"] != "", df["flags"] + "|" + fl, fl), df["flags"])
    return df


def load_labels(cfg, tree: Tree, df: pd.DataFrame, dirs):
    """Returns (leaf index per row or NaN, include flag per row or NaN, {"supplier"/"item": golden key per row}).
    Missing file -> no labels."""
    lc = cfg.get("labels") or {}
    leaf = pd.Series(np.nan, index=df.index); inc = pd.Series(np.nan, index=df.index)
    if not lc.get("file"):
        return leaf, inc, {}
    try:
        path = _find(lc["file"], *dirs)
    except FileNotFoundError:
        log.warning("labels file %s not found: running WITHOUT labels (tree-only classification)", lc["file"])
        return leaf, inc, {}
    lab = read_csv(path).drop_duplicates(lc["id_column"]).set_index(lc["id_column"])
    index = tree.index()
    cols = lc.get("levels") or tree.levels
    cols_data = [(lab[c].fillna("").astype(str).str.strip().tolist() if c in lab.columns else [""] * len(lab)) for c in cols]
    paths = [tuple(p) + ("",) * (tree.depth - len(cols)) for p in zip(*cols_data)]
    li = pd.Series([index.get(p, np.nan) for p in paths], index=lab.index)
    # labels found in the reference folder describe the reference records only: never let them land on a
    # new record that happens to reuse an id
    from_ref = len(dirs) > 1 and dirs[1] and Path(path).parent.resolve() == Path(dirs[1]).resolve() \
        and Path(dirs[0]).resolve() != Path(dirs[1]).resolve()
    rid = df["rid"].where(df["ref"]) if from_ref else df["rid"]
    leaf = rid.map(li)
    if lc.get("include_column") and lc["include_column"] in lab.columns:
        inc = rid.map(pd.to_numeric(lab[lc["include_column"]], errors="coerce"))
    gold = {}
    for what in ("supplier", "item"):
        col = lc.get(f"{what}_key_column")
        if col and col in lab.columns:
            gold[what] = rid.map(lab[col])
    unknown = int(sum(1 for p, v in zip(paths, li) if any(p) and pd.isna(v)))
    log.info("labels: %d records labelled with a category (%d label paths not in the tree, ignored)", int(leaf.notna().sum()), unknown)
    return leaf, inc, gold


def run(cfg: dict, data_dir, out_path, reference_dir=None, use_labels=True, llm=None, audit=True) -> pd.DataFrame:
    t0 = time.time()
    reference_dir = reference_dir or cfg.get("reference_dir")
    if reference_dir and not Path(reference_dir).exists():
        log.warning("reference_dir %s does not exist: running without reference data", reference_dir)
        reference_dir = None
    dirs = [data_dir, reference_dir]
    tree = Tree(_find(cfg["taxonomy"]["file"], *dirs, columns=cfg["taxonomy"].get("levels")), cfg["taxonomy"].get("levels"))
    df = ingest(cfg, data_dir, reference_dir)
    df = clean_money(df, cfg, dirs)
    df["status_class"] = exclusions.status_class(df["status_raw"], load_status_map(cfg))
    leaf_lab, inc_lab, gold = load_labels(cfg, tree, df, dirs) if use_labels else (
        pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index), {})
    df["supplier_key"] = dedupe.supplier_keys(df, cfg.get("supplier_dedupe"), gold.get("supplier"))
    dup_of, how, links = match.find_duplicates(df, cfg)
    df["item_key"] = dedupe.item_keys(df, links, cfg.get("item_dedupe"), gold.get("item"))
    # records without supplier/item text inherit them from the record they duplicate
    rid_pos = pd.Series(df.index, index=df["rid"])
    rid_pos = rid_pos[~rid_pos.index.duplicated()]
    src = dup_of.map(rid_pos)
    for col in ("supplier_key", "item_key"):
        inherit = src.notna() & ((df[col] == "") | (how.isin(["PO+AMOUNT", "PO+AMOUNT(OTHER_TAX_RATE)", "SUPPLIER+AMOUNT"])))
        df.loc[inherit, col] = df[col].reindex(src[inherit].astype(int)).to_numpy()

    np_rule = exclusions.non_procurement(df, cfg, tree.name_words())
    if (cfg.get("labels") or {}).get("learn_exclusions", True):
        learned = exclusions.learned_exclusions(df, inc_lab)
        np_rule = np_rule.where(np_rule != "", learned)

    reason = pd.Series("", index=df.index, dtype="object"); rconf = pd.Series(np.nan, index=df.index)
    def put(mask, code, conf):
        m = mask & (reason == "")
        reason[m] = code; rconf[m] = conf
    dconf = how.map({"EXACT_ROW": 0.97, "REPEAT_EXPORT": 0.95, "PO+AMOUNT": 0.97, "PO+AMOUNT(OTHER_TAX_RATE)": 0.8, "SUPPLIER+AMOUNT": 0.85})
    # brief: a purchase present in both systems is counted in the system of record, the other copy is DUPLICATE
    put((dup_of != "") & how.isin(["PO+AMOUNT", "PO+AMOUNT(OTHER_TAX_RATE)", "SUPPLIER+AMOUNT"]), "DUPLICATE", dconf)
    put(df["status_class"] == "CANCELLED", "CANCELLED", 0.99)
    put(df["status_class"].isin(["NOT_APPROVED", "UNKNOWN"]), "NOT_APPROVED", np.where(df["status_class"] == "UNKNOWN", 0.6, 0.99))
    put(inc_lab == 0, "NON_PROCUREMENT", 0.97)
    put((np_rule != "") & ~(inc_lab == 1), "NON_PROCUREMENT", np.where(np_rule.str.startswith("learned"), 0.85, 0.9))
    put(dup_of != "", "DUPLICATE", dconf)
    df["reason"], df["np_rule"], df["dup_of"], df["match_method"] = reason, np_rule, dup_of, how

    # classify everything that is included (reference rows with labels are the training set)
    todo = (reason == "") | df["ref"]
    cls = classify(df[todo], tree, cfg, leaf_lab[todo], llm=llm)
    run_rows = df[~df["ref"]]
    inc = (run_rows["reason"] == "")
    out = pd.DataFrame({"record_id": run_rows["rid"], "include": inc.astype(int), "exclusion_reason": run_rows["reason"],
                        "supplier_key": run_rows["supplier_key"], "item_key": run_rows["item_key"]})
    spend = run_rows["net"].where(inc, 0.0)
    unparsed = inc & run_rows["net"].isna()
    if unparsed.any():
        log.warning("%d included records have no usable amount (spend set to 0, flagged in the audit file)", int(unparsed.sum()))
    out["spend"] = spend.fillna(0.0).round(2)
    leaf = cls["leaf"].reindex(run_rows.index)
    for k in range(4):
        names = [tree.leaves[int(l)][k] if (pd.notna(l) and k < tree.depth) else "" for l in leaf]
        out[f"l{k + 1}"] = np.where(inc, names, "")
    conf = cls["confidence"].reindex(run_rows.index).where(inc, rconf.reindex(run_rows.index))
    conf = conf.where(~unparsed, np.minimum(conf, 0.3))
    out["confidence"] = conf.fillna(0.5).round(3)
    out = out[OUT_COLS]
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    if audit:
        a = pd.DataFrame({"record_id": run_rows["rid"], "source": run_rows["source"], "include": inc.astype(int),
                          "exclusion_reason": run_rows["reason"], "duplicate_of": run_rows["dup_of"],
                          "match_method": run_rows["match_method"], "non_procurement_rule": run_rows["np_rule"],
                          "status_raw": run_rows["status_raw"], "amount_raw": run_rows["amount_raw"],
                          "currency": run_rows["currency"], "tax_rate": run_rows["tax_rate"], "fx_rate_used": run_rows["fx_used"],
                          "net_reporting": run_rows["net"].round(2), "date_parsed": run_rows["date"].dt.strftime("%Y-%m-%d"),
                          "flags": run_rows["flags"], "class_method": cls["method"].reindex(run_rows.index).where(inc, ""),
                          "second_choice": ["" if (pd.isna(l) or not i) else " > ".join(x for x in tree.leaves[int(l)] if x)
                                            for l, i in zip(cls["second_leaf"].reindex(run_rows.index), inc)]})
        audit_dir = Path(audit if isinstance(audit, (str, Path)) else "audit")
        audit_dir.mkdir(parents=True, exist_ok=True)
        a.to_csv(audit_dir / (out_path.stem + "_audit.csv"), index=False, encoding="utf-8")
    secs = time.time() - t0
    summary = out.groupby(out["exclusion_reason"].replace("", "INCLUDED")).agg(records=("include", "size"), spend=("spend", "sum"))
    log.info("done in %.1fs: %d records, total spend %s %.2f\n%s", secs, len(out), cfg["reporting_currency"], out["spend"].sum(), summary.to_string())
    flags = run_rows["flags"][run_rows["flags"] != ""].str.split("|").explode().value_counts()
    if len(flags):
        log.info("data-quality flags (see audit file):\n%s", flags.to_string())
    log.info("wrote %s", out_path)
    return out
