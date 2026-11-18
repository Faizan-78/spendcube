"""Classify lines into a category tree that is read at run time.

Three sources of evidence, combined in the same way whatever the company:

  1. LEXICAL  - similarity between the line (description, account name, context) and each category,
                using the category names themselves, a generic procurement lexicon attached to the
                categories by trigger words, and an abbreviation matcher (CNVY -> conveyor,
                STNEDST -> "stone dust") so that SAP-style short texts can hit tree words.
  2. LABELS   - analyst labels, when the company has any (reference data or seed labels).
  3. LEARNED  - a linear model over text + supplier + account + item features, trained on (2) and/or on
                the confident predictions of (1) (self-training). It carries evidence to lines whose own
                text says nothing ("Parts", blank) through their supplier, account and item.

An optional LLM can add labels for the highest-spend distinct lines (see llm.py); it is off by default.
No category name appears in this file.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from .textnorm import skeleton, stem, strip_accents, words

log = logging.getLogger("spendcube")
RES = Path(__file__).parent / "resources"
STOP = set("and or of the for to in on with by per a an und der die das de la le el los las et du des da do e y "
           "other general misc miscellaneous various related unclassified nec".split())


class Tree:
    """Category tree: one leaf per distinct row of the tree file; levels may be ragged."""

    def __init__(self, path, levels=None):
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", sep=None, engine="python")
        levels = levels or list(raw.columns)
        self.levels = levels
        rows = raw[levels].apply(lambda c: c.str.strip()).drop_duplicates()
        rows = rows[(rows != "").any(axis=1)]
        self.leaves = [tuple(r) for r in rows.itertuples(index=False, name=None)]
        self.depth = len(levels)
        if not self.leaves:
            raise ValueError(f"category tree {path} is empty")

    def __len__(self):
        return len(self.leaves)

    def name_words(self) -> set:
        return {w for leaf in self.leaves for name in leaf for w in words(name)}

    def index(self) -> dict:
        return {leaf: i for i, leaf in enumerate(self.leaves)}

    def parents(self, level: int) -> np.ndarray:
        """Group id of each leaf at the given depth (0-based)."""
        keys = [leaf[: level + 1] for leaf in self.leaves]
        ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        return np.array([ids[k] for k in keys])


def _phrase_in(trigger: str, stems: list[str]) -> bool:
    t = [stem(x) for x in strip_accents(trigger).lower().split()]
    n = len(t)
    return any(stems[i:i + n] == t for i in range(len(stems) - n + 1))


class LexicalScorer:
    """Scores every (line, leaf) pair from names + lexicon + abbreviations."""

    def __init__(self, tree: Tree, use_lexicon=True, lexicon_path=None, acronyms=True):
        self.tree = tree
        concepts = []
        if use_lexicon:
            concepts = yaml.safe_load(Path(lexicon_path or RES / "lexicon.yaml").read_text(encoding="utf-8"))["concepts"]
        vocab, rows = {}, []

        def vid(w):
            return vocab.setdefault(w, len(vocab))

        self.n_concepts_used = 0
        for leaf in tree.leaves:
            names = [n for n in leaf if n]
            weights = defaultdict(float)
            distinct = list(dict.fromkeys(names))
            for depth_from_leaf, name in enumerate(reversed(distinct)):
                lw = [1.0, 0.5, 0.3, 0.2][min(depth_from_leaf, 3)]
                st = [w for w in words(name)]
                for w in st:
                    if w not in STOP:
                        weights[w] = max(weights[w], lw)
                content = [w for w in st if w not in STOP]
                for a, b in zip(content, content[1:]):  # compounds: "roof bolt" -> roofbolt
                    weights[a + b] = max(weights[a + b], lw)
                if acronyms and len(content) >= 3:  # acronyms: "can body stock" -> cbs
                    acro = "".join(w[0] for w in content)
                    weights[acro] = max(weights[acro], 0.9 * lw)
                for c in concepts:
                    if any(_phrase_in(t, st) for t in c["triggers"]):
                        self.n_concepts_used += 1
                        for term in c["terms"]:
                            tw = [w for w in words(term) if w not in STOP]
                            for w in tw:
                                weights[w] = max(weights[w], 0.6 * lw / (1 if len(tw) == 1 else 1.3))
                            if len(tw) > 1:
                                weights["".join(tw)] = max(weights["".join(tw)], 0.6 * lw)
            rows.append({vid(w): v for w, v in weights.items()})
        self.vocab = vocab
        N = sp.lil_matrix((len(tree), len(vocab)), dtype=np.float32)
        for i, r in enumerate(rows):
            for j, v in r.items():
                N[i, j] = v
        N = N.tocsr()
        df_t = np.asarray((N > 0).sum(axis=0)).ravel()
        self.idf = (np.log((len(tree) + 1) / (df_t + 1)) + 1.0).astype(np.float32)
        N = N.multiply(self.idf).tocsr()
        norms = np.sqrt(np.asarray(N.multiply(N).sum(axis=1)).ravel()) + 1e-9
        self.N = sp.diags(1.0 / norms) @ N
        # abbreviation index: skeleton prefixes and word prefixes of the vocabulary
        self._skel, self._pref = defaultdict(set), defaultdict(set)
        for w in vocab:
            if len(w) < 4:
                continue
            sk = skeleton(w)
            for k in range(3, len(sk) + 1):
                self._skel[sk[:k]].add(w)
            for k in range(4, len(w)):
                self._pref[w[:k]].add(w)

    def _expand(self, tok: str) -> list[tuple[int, float]]:
        v = self.vocab
        if tok in v:
            return [(v[tok], 1.0)]
        out = {}
        for w in self._pref.get(tok, ()):  # truncation: "maintenan" -> maintenance
            out[w] = 0.8
        if len(tok) >= 3:
            vowels = sum(c in "aeiou" for c in tok[1:])
            if vowels <= max(1, len(tok) // 5):  # looks like a consonant abbreviation
                sk = skeleton(tok)
                if len(sk) >= 3:
                    for w in self._skel.get(sk, ()):
                        if len(w) > len(tok):
                            out.setdefault(w, 0.6 if len(sk) >= 4 else 0.4)
        if not out:
            return []
        share = 1.0 / (1 + 0.5 * (len(out) - 1))  # ambiguous abbreviations count less
        return [(v[w], s * share) for w, s in out.items()]

    def score(self, main_text: list[str], extra_text: list[str]) -> np.ndarray:
        """Dense (n_docs, n_leaves) score matrix. extra_text (account names, context) counts 0.7."""
        tok_index, T_rows = {}, []

        def tid(t):
            if t not in tok_index:
                tok_index[t] = len(tok_index)
                T_rows.append(self._expand(t))
            return tok_index[t]

        r, c, d = [], [], []
        for i, (a, b) in enumerate(zip(main_text, extra_text)):
            seen = {}
            for text, w in ((a, 1.0), (b, 0.7)):
                toks = [t for t in words(text) if t not in STOP]
                for t in toks + [x + y for x, y in zip(toks, toks[1:])]:
                    j = tid(t)
                    seen[j] = max(seen.get(j, 0.0), w)
            for j, w in seen.items():
                r.append(i); c.append(j); d.append(w)
        D = sp.csr_matrix((d, (r, c)), shape=(len(main_text), max(len(tok_index), 1)), dtype=np.float32)
        tr, tc, td = [], [], []
        for j, pairs in enumerate(T_rows):
            for vi, w in pairs:
                tr.append(j); tc.append(vi); td.append(w)
        T = sp.csr_matrix((td, (tr, tc)), shape=(max(len(tok_index), 1), len(self.vocab)), dtype=np.float32)
        L = (D @ T).multiply(self.idf).tocsr()
        return np.asarray((L @ self.N.T).todense())


def _features(docs: pd.DataFrame, fit_idx=None):
    """Sparse features for the learned model. docs has text, ctx, cats columns."""
    text = (docs["text"] + " ¦ " + docs["ctx"]).str.lower()
    v_word = TfidfVectorizer(analyzer="word", token_pattern=r"[^\W\d_]{2,}|\d+[^\W\d_]+|[^\W\d_]+\d+", ngram_range=(1, 2),
                             min_df=1, sublinear_tf=True, max_features=300000, dtype=np.float32)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                             max_features=400000, dtype=np.float32)
    v_cat = TfidfVectorizer(analyzer=lambda s: s.split(), use_idf=False, norm="l2", dtype=np.float32)
    Xw = v_word.fit_transform(text)
    Xc = v_char.fit_transform(docs["text"].str.lower())
    try:
        Xk = v_cat.fit_transform(docs["cats"])
    except ValueError:
        Xk = sp.csr_matrix((len(docs), 0), dtype=np.float32)
    return sp.hstack([Xw, 0.7 * Xc, 1.0 * Xk]).tocsr()


def _fit_predict(X, y_idx, y, n_classes, alpha=2e-6, sample_weight=None):
    """Train on rows y_idx with labels y; return probabilities for all rows over n_classes."""
    classes = np.unique(y)
    P = np.zeros((X.shape[0], n_classes), dtype=np.float32)
    if len(classes) == 1:
        P[:, classes[0]] = 1.0
        return P
    clf = SGDClassifier(loss="modified_huber", alpha=alpha, max_iter=40, tol=1e-4, random_state=0)
    clf.fit(X[y_idx], y, sample_weight=sample_weight)
    P[:, clf.classes_] = clf.predict_proba(X).astype(np.float32)
    return P


def _neighbour_votes(docs: pd.DataFrame, final: np.ndarray, evidenced: np.ndarray, weight: np.ndarray):
    """For each doc, the spend-weighted mean prediction of evidenced docs sharing its item, else supplier,
    else account. A line that says nothing ("Parts", a coil spec) is most likely bought like its siblings."""
    out = np.zeros_like(final)
    how = np.full(len(docs), "", dtype=object)
    w = (np.sqrt(np.maximum(weight, 0)) + 1.0) * evidenced
    for col in ("item", "sup", "gl"):
        keys = docs[col].fillna("").astype(str).to_numpy()
        codes, uniq = pd.factorize(keys)
        sums = np.zeros((len(uniq), final.shape[1]), dtype=np.float64)
        np.add.at(sums, codes, final * w[:, None])
        tot = np.bincount(codes, weights=w, minlength=len(uniq))
        valid = (tot > 0) & (np.array([u not in ("", "|") for u in uniq]))
        todo = (how == "") & valid[codes]
        out[todo] = sums[codes[todo]] / tot[codes[todo], None]
        how[todo] = {"item": "item", "sup": "supplier", "gl": "account"}[col]
    return out, how


def classify(df: pd.DataFrame, tree: Tree, cfg: dict, labels: pd.Series | None = None, llm=None) -> pd.DataFrame:
    """df: canonical frame (run + reference rows). labels: leaf index per row label (NaN = unlabelled).

    Returns a frame indexed like df with leaf (int), confidence (0-1) and method.
    """
    ccfg = cfg.get("classify") or {}
    n = len(tree)
    labels = labels if labels is not None else pd.Series(np.nan, index=df.index)
    cats = ("sup=" + df["supplier_key"].str.replace(r"\s+", "_", regex=True)).where(df["supplier_key"] != "", "")
    cats = cats + np.where(df["gl_account"] != "", " gl=" + df["gl_account"], "")
    cats = cats + np.where(df["item_key"] != "", " item=" + df["item_key"].str.replace(r"\s+", "_", regex=True), "")
    cats = (cats + " " + df["ctx_cat"]).str.strip()
    ctx = (df["gl_desc"] + " " + df["ctx_text"]).str.strip()
    key = df["desc"] + "\x1f" + ctx + "\x1f" + cats
    codes, uniq = pd.factorize(key, sort=False)
    first = pd.Series(np.arange(len(df)), index=codes).groupby(level=0).first().to_numpy()
    docs = pd.DataFrame({"text": df["desc"].to_numpy()[first], "ctx": ctx.to_numpy()[first], "cats": cats.to_numpy()[first],
                         "item": df["item_key"].to_numpy()[first], "sup": df["supplier_key"].to_numpy()[first],
                         "gl": (df["gl_account"] + "|" + df["gl_desc"]).to_numpy()[first]})
    weight = np.bincount(codes, weights=df["net"].abs().fillna(0).to_numpy(), minlength=len(docs))
    log.info("classification: %d lines -> %d distinct (text, context, supplier, account, item) combinations; %d leaves",
             len(df), len(docs), n)

    # 1. lexical evidence
    lex = LexicalScorer(tree, use_lexicon=ccfg.get("use_lexicon", True), lexicon_path=ccfg.get("lexicon"),
                        acronyms=ccfg.get("acronyms", True))
    S = lex.score(docs["text"].tolist(), docs["ctx"].tolist())
    top = S.argmax(axis=1)
    s1 = S[np.arange(len(docs)), top]
    S2 = S.copy(); S2[np.arange(len(docs)), top] = -1
    s2 = S2.max(axis=1)
    margin = np.where(s1 > 0, (s1 - s2) / np.maximum(s1, 1e-9), 0.0)

    # 2. labels per distinct doc (majority of its labelled lines)
    lab = pd.DataFrame({"doc": codes, "y": labels.to_numpy()}).dropna()
    doc_label = np.full(len(docs), -1, dtype=int)
    if len(lab):
        maj = lab.groupby("doc")["y"].agg(lambda s: s.value_counts().index[0]).astype(int)
        doc_label[maj.index.to_numpy()] = maj.to_numpy()
    n_lab = int((doc_label >= 0).sum())

    # optional LLM labels for the highest-spend distinct lines that have no analyst label
    llm_label = np.full(len(docs), -1, dtype=int)
    if llm is not None:
        cand = np.argsort(-weight)
        cand = [i for i in cand if doc_label[i] < 0 and (docs.at[i, "text"] or docs.at[i, "ctx"])][: llm.max_items]
        got = llm.label([(docs.at[i, "text"], docs.at[i, "ctx"]) for i in cand], tree)
        for i, g in zip(cand, got):
            if g is not None and 0 <= g < n:
                llm_label[i] = g
        log.info("classification: LLM labelled %d distinct lines", int((llm_label >= 0).sum()))

    X = _features(docs)
    method = np.full(len(docs), "", dtype=object)
    supervised = n_lab >= ccfg.get("min_labels_for_supervised", 200)
    if supervised:
        tr = np.where(doc_label >= 0)[0]
        P = _fit_predict(X, tr, doc_label[tr], n, alpha=ccfg.get("alpha", 2e-6))
        Sn = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-9)
        has_lex = (s1 >= ccfg.get("min_lex_score", 0.08))[:, None]
        final = np.where(has_lex, 0.85 * P + 0.15 * Sn, P)
        conf = final.max(axis=1)
        method[:] = "MODEL(labels)"
    else:
        # self-training from confident lexical matches (and LLM labels / the few analyst labels)
        strong = (s1 >= ccfg.get("pseudo_min_score", 0.18)) & (margin >= ccfg.get("pseudo_min_margin", 0.25))
        pseudo = np.where(strong, top, -1)
        pseudo = np.where(llm_label >= 0, llm_label, pseudo)
        pseudo = np.where(doc_label >= 0, doc_label, pseudo)
        tr = np.where(pseudo >= 0)[0]
        Sn = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-9)
        sharp = np.clip(s1 / 0.5, 0, 1)[:, None]
        if len(tr) >= 20 and len(np.unique(pseudo[tr])) >= 2:
            sw = np.where((doc_label[tr] >= 0) | (llm_label[tr] >= 0), 3.0, np.clip(s1[tr] * (0.5 + margin[tr]) * 2, 0.2, 1.5))
            # cap very frequent pseudo-classes so one easy category cannot swamp the model
            cnt = np.bincount(pseudo[tr], minlength=n)
            cap = max(50.0, 4 * np.median(cnt[cnt > 0]))
            sw = sw * np.minimum(1.0, cap / cnt[pseudo[tr]])
            P = _fit_predict(X, tr, pseudo[tr], n, alpha=ccfg.get("alpha_selftrain", 2e-5), sample_weight=sw)
            lex_part = Sn ** 1.0
            alpha = 0.25 + 0.5 * sharp  # trust the lexical match more when it is strong
            final = alpha * lex_part + (1 - alpha) * P
            weak = s1 < ccfg.get("min_lex_score", 0.08)
            nb, nb_how = _neighbour_votes(docs, final, ~weak, weight)
            has_nb = weak & (nb.sum(axis=1) > 0) & bool(ccfg.get("neighbours", False))
            final = np.where(weak[:, None], P, final)
            mix = np.array([{"item": 0.7, "supplier": 0.35, "account": 0.25}.get(h, 0.0) for h in nb_how])[:, None]
            final[has_nb] = (mix * nb + (1 - mix) * P)[has_nb]
            method[:] = np.where(~weak, "LEXICAL+SELF_TRAINED", "SELF_TRAINED(context only)")
            method[has_nb] = ["NEIGHBOURS(" + h + ")" for h in nb_how[has_nb]]
        else:
            log.warning("classification: almost no line matches the tree vocabulary; falling back to lexical scores only")
            final = Sn.copy()
            method[:] = "LEXICAL"
        f1 = final.max(axis=1)
        fs = np.sort(final, axis=1)
        fm = (fs[:, -1] - fs[:, -2]) / np.maximum(fs[:, -1], 1e-9) if n > 1 else np.ones(len(docs))
        conf = np.clip(0.9 * f1 * (0.6 + 0.4 * fm) + 0.2 * sharp[:, 0] * margin, 0.02, 0.9)
        for src, name in ((llm_label, "LLM"), (doc_label, "LABEL")):
            hit = src >= 0
            final[hit] = 0; final[hit, src[hit]] = 1.0
            conf = np.where(hit, 0.9, conf); method[hit] = name
    if supervised:
        hit = doc_label >= 0
        final[hit] = 0.1 * final[hit]
        final[hit, doc_label[hit]] += 0.9
        conf = np.where(hit, np.maximum(conf, 0.97), conf)
        method[hit] = "LABEL"
    nothing = final.max(axis=1) <= 0
    if nothing.any():
        prior = np.bincount(final.argmax(axis=1)[~nothing], minlength=n) if (~nothing).any() else np.ones(n)
        final[nothing, prior.argmax()] = 1e-6
        conf = np.where(nothing, 0.02, conf)
        method[nothing] = "FALLBACK(most common leaf)"
    leaf = final.argmax(axis=1)
    out = pd.DataFrame({"leaf": leaf[codes], "confidence": np.round(np.clip(conf[codes], 0, 1), 3),
                        "method": method[codes]}, index=df.index)
    order2 = np.argsort(final, axis=1)[:, -2] if n > 1 else leaf
    out["second_leaf"] = order2[codes]
    return out
