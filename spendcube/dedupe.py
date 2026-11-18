"""Supplier and item de-duplication.

Both use the same recipe:
  1. a normalised comparison key (case, punctuation, spacing, legal forms, OCR confusions removed);
  2. hard links: same id <-> same name (unless the id is a catch-all used for many names);
  3. cautious fuzzy links for typing/OCR variants, never between two "trusted" (master-data) keys
     and never when the difference changes a number;
  4. union-find, then a readable key per cluster.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .textnorm import compact, levenshtein, ocr_fold, strip_accents, text_key

log = logging.getLogger("spendcube")

LEGAL_FORMS = set("""PTY LTD LIMITED CC INC INCORPORATED LLC LLP LP PLC CORP CORPORATION CO COMPANY GMBH AG KG
MBH UG SA SAS SARL SL SLU SRL SPA BV NV AB AS ASA OY OYJ APS PVT PRIVATE PTE SDN BHD LTDA EIRELI SAC CV KFT SRO
DOO OOO ZAO TBK PT THE AND DE SP ZOO KK EPE EOOD JSC PJSC LLLP TRADING HOLDINGS GROUP INTERNATIONAL INTL""".split())
_SOFT = {"TRADING", "HOLDINGS", "GROUP", "INTERNATIONAL", "INTL"}  # dropped only when something else remains


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _apply_golden(uf: UnionFind, node: pd.Series, golden, what: str):
    """Nodes that the client's golden key puts together are one entity; new records reach them by name/code."""
    if golden is None:
        return
    g = golden.reindex(node.index)
    m = g.notna() & (g.astype(str) != "") & (node != "")
    for n, k in zip(node[m], g[m].astype(str)):
        uf.union("golden:" + k, n)
    if m.any():
        log.info("%s de-dup: guided by %d labelled records (%d golden keys)", what, int(m.sum()), g[m].nunique())


def supplier_name_key(name) -> str:
    if name is None:
        return ""
    s = strip_accents(str(name)).upper()
    s = re.sub(r"\b([A-Z])\.(?=[A-Z]\b)", r"\1", s)  # S.A. -> SA., B.V. -> BV.
    toks = re.findall(r"[0-9A-Z]+", s)
    hard = [t for t in toks if t not in LEGAL_FORMS or t in _SOFT]
    core = [t for t in hard if t not in _SOFT] or hard
    return ocr_fold("".join(core or toks))


def number_signature(raw) -> tuple:
    """Multi-digit numbers of a raw text ("1800MM" -> ('1800',)). Single digits are left out: OCR turns
    letters into lone digits (R0OF, B1T), while sizes and part numbers are longer runs."""
    s = re.sub(r"(?<=\d)[,.](?=\d)", "", str(raw or ""))
    return tuple(sorted(re.findall(r"\d{2,}", s)))


def _digits(s: str) -> list[str]:
    return sorted(re.findall(r"[234679]", s))


def fuzzy_links(keys: list[str], trusted: set[str], max_d, allow_substitution, min_sim=0.55, top_k=6, signature=None):
    """Candidate pairs by character n-gram similarity, verified by edit distance.

    Returns a list of (key_a, key_b). max_d(len) -> allowed number of edits.
    signature: optional key -> tuple of multi-digit numbers in the raw text; keys whose numbers differ never link.
    """
    keys = [k for k in keys if len(k) >= 5]
    if len(keys) < 2:
        return []
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), dtype=np.float32, sublinear_tf=True)
    X = vec.fit_transform(keys).tocsr()
    links, step = [], 2000
    for start in range(0, len(keys), step):
        S = (X[start:start + step] @ X.T).tocsr()
        for i in range(S.shape[0]):
            lo, hi = S.indptr[i], S.indptr[i + 1]
            cols, vals = S.indices[lo:hi], S.data[lo:hi]
            if len(cols) > top_k + 1:
                sel = np.argpartition(-vals, top_k)[: top_k + 1]
                cols, vals = cols[sel], vals[sel]
            a = keys[start + i]
            for j, v in zip(cols, vals):
                if j <= start + i or v < min_sim:
                    continue
                b = keys[j]
                if a in trusted and b in trusted:
                    continue
                md = max_d(min(len(a), len(b)))
                if md <= 0:
                    continue
                d = levenshtein(a, b, md)
                if d == 0 or d > md:
                    continue
                indel_only = d == abs(len(a) - len(b))
                if not indel_only and not (allow_substitution and d == 1):
                    continue
                if _digits(a) != _digits(b):
                    continue
                if signature is not None and signature.get(a, ()) != signature.get(b, ()):
                    continue
                links.append((a, b))
    return links


def _label_clusters(uf: UnionFind, node_of_row: pd.Series, display: pd.Series) -> pd.Series:
    """Readable, unique key per cluster: the most common display string of its rows."""
    roots = node_of_row.map(lambda n: uf.find(n) if n else "")
    out = pd.Series("", index=node_of_row.index, dtype="object")
    used = {}
    disp = display.fillna("").astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    for root, idx in sorted(roots[roots != ""].groupby(roots[roots != ""]).groups.items(), key=lambda kv: str(kv[0])):
        names = [d for d in disp.loc[idx] if d]
        label = Counter(names).most_common(1)[0][0] if names else str(root).lower()
        if used.setdefault(label, root) != root:
            n = 2
            while f"{label} #{n}" in used:
                n += 1
            label = f"{label} #{n}"
            used[label] = root
        out.loc[idx] = label
    return out


def supplier_keys(df: pd.DataFrame, cfg: dict | None = None, golden: pd.Series | None = None) -> pd.Series:
    """golden: the client's own supplier key where labels give one (guides grouping; NaN elsewhere)."""
    cfg = cfg or {}
    nk = df["supplier_name"].map(supplier_name_key)
    sid = df["supplier_id"].fillna("").astype(str)
    uf = UnionFind()
    names_per_id = defaultdict(set)
    for k, i in zip(nk, sid):
        if k and i:
            names_per_id[i].add(k)
    catch_all = {i for i, ks in names_per_id.items() if len(ks) > cfg.get("max_names_per_id", 3)}
    if catch_all:
        log.info("supplier de-dup: %d supplier ids used for many different names are not used as links", len(catch_all))
    for i, ks in names_per_id.items():
        if i not in catch_all:
            for k in ks:
                uf.union("id:" + i, "n:" + k)
    counts = Counter(k for k in nk if k)
    trusted = {k for i, ks in names_per_id.items() if i not in catch_all for k in ks}
    if cfg.get("fuzzy", True):
        md = lambda n: 0 if n < 6 else (1 if n < 14 else 2)
        links = fuzzy_links(sorted(counts), trusted, md, allow_substitution=cfg.get("allow_substitution", True))
        links = [(a, b) for a, b in links if not (levenshtein(a, b, 2) != abs(len(a) - len(b)) and min(len(a), len(b)) < 8)]
        for a, b in links:
            uf.union("n:" + a, "n:" + b)
        log.info("supplier de-dup: %d fuzzy name links", len(links))
    node = pd.Series(np.where(nk != "", "n:" + nk, np.where(sid != "", "id:" + sid, "")), index=df.index)
    _apply_golden(uf, node, golden, "supplier")
    display = df["supplier_name"].where(df["supplier_name"].fillna("") != "", sid)
    display = display.map(lambda s: re.sub(r"\s+", " ", str(s)).strip())
    return _label_clusters(uf, node, display)


def item_keys(df: pd.DataFrame, links=(), cfg: dict | None = None, golden: pd.Series | None = None) -> pd.Series:
    """links: pairs of row labels (index values) known to be the same purchase (cross-system matches)."""
    cfg = cfg or {}
    tk = df["item_desc"].map(text_key)
    tk = tk.where(tk.str.len() >= cfg.get("min_text_len", 3), "")
    code = df["item_code"].fillna("").astype(str)
    texts_per_code = defaultdict(set)
    for c, t in zip(code, tk):
        if c and t:
            texts_per_code[c].add(t)
    n_generic = cfg.get("generic_code_texts", 4)
    generic = {c for c, ts in texts_per_code.items() if len(ts) >= n_generic}
    generic |= set(cfg.get("generic_codes", []))
    if generic:
        log.info("item de-dup: %d item codes carry many different descriptions (free-text/dummy codes); "
                 "their lines are keyed by description (policy=%s)", len(generic), cfg.get("generic_code_policy", "split"))
    lump = cfg.get("generic_code_policy", "split") == "lump"
    uf = UnionFind()
    trusted = set()
    for c, ts in texts_per_code.items():
        if c in generic and not lump:
            continue
        for t in ts:
            if c not in generic:
                uf.union("c:" + c, "t:" + t)
                trusted.add(t)
    if cfg.get("fuzzy", True):
        md = lambda n: 0 if n < 8 else (1 if n < 16 else (2 if n < 30 else 3))
        fold2 = {}
        for t in set(tk) - {""}:
            fold2.setdefault(t.replace("C", "E"), t)
        tr2 = {t.replace("C", "E") for t in trusted}
        for t in set(tk) - {""}:
            uf.union("t:" + t, "t:" + fold2[t.replace("C", "E")])
        sig_raw = {}
        for t, raw in zip(tk, df["item_desc"]):
            if t:
                sig_raw.setdefault(t.replace("C", "E"), Counter())[number_signature(raw)] += 1
        sig = {k: c.most_common(1)[0][0] for k, c in sig_raw.items()}
        fl = fuzzy_links(sorted(fold2), tr2, md, allow_substitution=False, signature=sig)
        for a, b in fl:
            uf.union("t:" + fold2[a], "t:" + fold2[b])
        log.info("item de-dup: %d fuzzy description links", len(fl))
    has_code = (code != "") & (~code.isin(generic) | lump)
    node = pd.Series(np.where(has_code, "c:" + code, np.where(tk != "", "t:" + tk, "")), index=df.index)
    from difflib import SequenceMatcher
    n_links = 0
    for a, b in links:  # same purchase in two systems: same item only if the two texts agree
        na, nb = node.get(a, ""), node.get(b, "")
        ta, tb = tk.get(a, ""), tk.get(b, "")
        if na and nb and ta and tb and number_signature(df.at[a, "item_desc"]) == number_signature(df.at[b, "item_desc"]) \
                and SequenceMatcher(None, ta, tb, autojunk=False).ratio() >= cfg.get("link_min_text_sim", 0.75):
            uf.union(na, nb)
            n_links += 1
    log.info("item de-dup: %d of %d cross-system matches also link item descriptions", n_links, len(links))
    _apply_golden(uf, node, golden, "item")
    display = df["item_desc"].where(df["item_desc"].fillna("") != "", code)
    keys = _label_clusters(uf, node, display)
    # nothing to go on (no usable code, no text): group by the code if there is one, else "unspecified"
    blank = keys == ""
    keys[blank] = np.where(code[blank] != "", code[blank].str.lower() + " (no description)", "unspecified")
    return keys
