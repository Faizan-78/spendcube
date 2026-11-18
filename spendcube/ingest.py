"""Read a company's files as described by its config and map them onto one canonical frame."""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .parsing import is_blank, parse_date, parse_number, parse_rate
from .textnorm import compact

log = logging.getLogger("spendcube")
RES = Path(__file__).parent / "resources"

CANON = ["rid", "source", "ref", "supplier_name", "supplier_id", "desc", "item_code", "qty", "amount_raw",
         "tax_rate", "currency", "fx_rate", "date", "alt_date", "status_raw", "po", "po_line", "inv",
         "gl_account", "gl_desc", "ctx_text", "ctx_cat", "row_hash", "flags", "order"]


def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_path"] = str(path)
    for key in ("company", "reporting_currency", "taxonomy", "sources"):
        if key not in cfg:
            raise ValueError(f"config {path}: missing required key '{key}'")
    return cfg


def read_csv(path: Path, opts: dict | None = None) -> pd.DataFrame:
    """Read every column as text; try common encodings; sniff the delimiter."""
    opts = dict(opts or {})
    for enc in [opts.pop("encoding", None), "utf-8-sig", "cp1252", "latin-1"]:
        if enc is None:
            continue
        try:
            if path.suffix.lower() in (".xlsx", ".xls"):
                return pd.read_excel(path, dtype=str, **opts)
            return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], sep=opts.pop("sep", None),
                               engine="python", encoding=enc, **opts)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def _norm_col(c) -> str:
    return re.sub(r"\W+", "", str(c)).lower()


def find_by_columns(folder: Path, wanted: list[str], exclude=()) -> Path | None:
    """When a configured file is absent (renamed in a new delivery), pick the data file in `folder` whose
    header holds the most of the expected columns; at least 60% of them must be there."""
    wanted = {_norm_col(c) for c in wanted if c}
    if not wanted or not Path(folder).is_dir():
        return None
    best, best_hits = None, 0
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() not in (".csv", ".txt", ".tsv", ".xlsx", ".xls") or f.name in exclude:
            continue
        try:
            head = pd.read_excel(f, nrows=0) if f.suffix.lower() in (".xlsx", ".xls") else \
                pd.read_csv(f, nrows=0, sep=None, engine="python", encoding="utf-8-sig")
        except Exception:
            continue
        hits = len(wanted & {_norm_col(c) for c in head.columns})
        if hits > best_hits:
            best, best_hits = f, hits
    if best is not None and best_hits >= 0.6 * len(wanted):
        return best
    return None


def _source_columns(src: dict) -> list[str]:
    cols = [src["id_column"]]
    for v in (src.get("fields") or {}).values():
        cols += v if isinstance(v, list) else [v]
    return [c for c in cols if c]


def _col(df: pd.DataFrame, name, source: str) -> pd.Series:
    """Fetch a configured column (case/space-insensitive); missing column -> all blank + warning."""
    if name is None:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    if name in df.columns:
        return df[name].astype("string")
    lookup = {re.sub(r"\W+", "", c).lower(): c for c in df.columns}
    hit = lookup.get(re.sub(r"\W+", "", str(name)).lower())
    if hit:
        return df[hit].astype("string")
    log.warning("source %s: configured column '%s' not found; treated as blank", source, name)
    return pd.Series(pd.NA, index=df.index, dtype="string")


def _join(df, names, source, sep=" | ") -> pd.Series:
    names = [names] if isinstance(names, str) else list(names or [])
    parts = [_col(df, n, source).fillna("").str.strip() for n in names]
    if not parts:
        return pd.Series("", index=df.index, dtype="object")
    out = parts[0].astype(object)
    for p in parts[1:]:
        p = p.astype(object)
        both = (out != "") & (p != "") & (out != p)
        out = np.where(both, out + sep + p, np.where(out == "", p, out))
        out = pd.Series(out, index=df.index, dtype="object")
    return out


def _clean_code(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    return s.mask(s.str.lower().isin({"nan", "none", "null", "n/a", "na", "-", "0"}), "")


def ingest_source(src: dict, data_dir: Path, ref: bool) -> pd.DataFrame | None:
    path = Path(data_dir) / src["file"]
    if not path.exists():
        if ref:
            return None
        alt = find_by_columns(Path(data_dir), _source_columns(src))
        if alt is None:
            raise FileNotFoundError(f"source {src['name']}: {path} not found, and no file in {data_dir} has its columns")
        log.warning("source %s: %s not found; using %s (its header matches the configured columns)", src["name"], src["file"], alt.name)
        path = alt
    raw = read_csv(path, src.get("read_options"))
    name, f = src["name"], src.get("fields", {})
    out = pd.DataFrame(index=raw.index)
    ids = _col(raw, src["id_column"], name).fillna("").str.strip()
    out["rid"] = src.get("record_prefix", "") + ids
    out["source"], out["ref"] = name, ref
    out["supplier_name"] = _col(raw, f.get("supplier_name"), name).fillna("").str.strip()
    out["supplier_id"] = _clean_code(_col(raw, f.get("supplier_id"), name))
    out["desc"] = _join(raw, f.get("description"), name)
    out["item_code"] = _clean_code(_col(raw, f.get("item_code"), name))
    out["item_desc"] = _join(raw, f["item_text"], name) if f.get("item_text") else out["desc"]
    out["qty"] = parse_number(_col(raw, f.get("quantity"), name), src.get("decimal", "auto"))
    amount_txt = _col(raw, f.get("amount"), name)
    out["amount_raw"] = parse_number(amount_txt, src.get("decimal", "auto"))
    amt = src.get("amount", {})
    out["tax_rate"] = parse_rate(_col(raw, f.get("tax_rate"), name), amt.get("default_tax_rate")) \
        if amt.get("basis", "net") == "gross" else 0.0
    cur = _col(raw, f.get("currency"), name).fillna("").str.strip().str.upper()
    out["currency"] = cur.astype(object) if f.get("currency") else ""
    out["fx_rate"] = parse_number(_col(raw, f.get("fx_rate"), name), src.get("decimal", "auto"))
    out["date"], dflag = parse_date(_col(raw, f.get("date"), name), src.get("date_formats"), src.get("dayfirst", "auto"))
    alt = f.get("alt_dates") or []
    out["alt_date"] = parse_date(_col(raw, alt[0], name), src.get("date_formats"))[0] if alt else pd.NaT
    out["status_raw"] = _col(raw, f.get("status"), name).fillna("").str.strip()
    po = _col(raw, f.get("po"), name).fillna("").str.strip()
    if src.get("po_pattern"):
        ext = po.str.extract(src["po_pattern"])
        out["po"] = ext[0].fillna("")
        out["po_line"] = _clean_code(ext[1]) if ext.shape[1] > 1 else ""
    else:
        out["po"] = po.map(compact).str.replace(r"^PO", "", regex=True).str.lstrip("0")
        out["po_line"] = ""
    if f.get("po_line"):
        pl = _clean_code(_col(raw, f["po_line"], name)).str.lstrip("0")
        out["po_line"] = np.where(pl != "", pl, out["po_line"])
    out["po_line"] = pd.Series(out["po_line"], index=raw.index).astype(str).str.lstrip("0")
    out["inv"] = _col(raw, f.get("invoice_number"), name).fillna("").map(compact)
    out["gl_account"] = _clean_code(_col(raw, f.get("gl_account"), name))
    out["gl_desc"] = _col(raw, f.get("gl_description"), name).fillna("").str.strip()
    cl = src.get("classify", {})
    out["ctx_text"] = _join(raw, cl.get("text"), name, sep=" ")
    cats = []
    for c in cl.get("categorical", []) or []:
        v = _clean_code(_col(raw, c, name))
        cats.append(np.where(v != "", c + "=" + v.str.replace(r"\s+", "_", regex=True), ""))
    out["ctx_cat"] = pd.Series([" ".join(x for x in row if x) for row in zip(*cats)] if cats else "", index=raw.index)
    # hash of the full business content of the row (everything except its id) for exact-duplicate detection
    body = raw.drop(columns=[c for c in [src["id_column"]] if c in raw.columns]).fillna("").astype(str)
    out["row_hash"] = [hashlib.md5("\x1f".join(r).encode("utf-8", "replace")).hexdigest() for r in body.itertuples(index=False, name=None)]
    flags = pd.Series("", index=raw.index, dtype="object")
    bad_amt = out["amount_raw"].isna()
    flags[bad_amt & ~is_blank(amount_txt)] = "AMOUNT_UNPARSEABLE"
    flags[bad_amt & is_blank(amount_txt)] = "AMOUNT_MISSING"
    if f.get("date"):
        flags = np.where(dflag != "", np.where(flags != "", flags + "|" + dflag, dflag), flags)
    out["flags"] = flags
    out["order"] = pd.to_numeric(ids.str.extract(r"(\d+)\D*$")[0], errors="coerce")
    out["_amount_basis"] = amt.get("basis", "net")
    out["_amount_currency"] = amt.get("currency", "column" if f.get("currency") else "reporting")
    dup = out["rid"].duplicated(keep=False) | (ids == "")
    if dup.any():
        log.warning("source %s: %d records have a blank or repeated id", name, int(dup.sum()))
    log.info("source %s (%s): %d rows from %s", name, "reference" if ref else "run", len(out), path.name)
    return out


def ingest(cfg: dict, data_dir, reference_dir=None) -> pd.DataFrame:
    frames = [ingest_source(s, Path(data_dir), False) for s in cfg["sources"]]
    if reference_dir and Path(reference_dir).resolve() != Path(data_dir).resolve():
        for s in cfg["sources"]:
            fr = ingest_source(s, Path(reference_dir), True)
            if fr is not None:
                frames.append(fr)
    df = pd.concat(frames, ignore_index=True)
    if df["ref"].any():  # a reference record never shadows a record of the run
        run_ids = set(df.loc[~df["ref"], "rid"])
        df = df[~(df["ref"] & df["rid"].isin(run_ids))].reset_index(drop=True)
    return df


def load_status_map(cfg: dict) -> dict:
    base = yaml.safe_load((RES / "status_map.yaml").read_text())
    for k, v in (cfg.get("status_map") or {}).items():
        base[k] = list(v) + [x for x in base.get(k, []) if x not in v]
    return base
