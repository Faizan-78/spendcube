# Approach

**Pipeline.** `run.py` → ingest by config (every column read as text; amounts, rates and dates parsed with
format detection; unparseable values flagged, not dropped) → net of tax (gross ÷ (1 + rate), blank rate = config
default) → reporting currency (line rate if it is within 3× of the monthly table, else the table; table month from
the invoice date, falling back to posting date when the date is missing or implausible; nearest month if needed)
→ supplier keys → count once → item keys → exclusions → classification → output + audit file. Nothing
company-specific is in code: files, columns, matching keys and extra rules live in `configs/*.yaml`, and the
category tree is a file read at run time.

**Counting once.** (1) Byte-identical rows (all columns but the id) are DUPLICATE. (2) A source's repeated
exports (same supplier, invoice number, net amount and quantity, with OCR noise allowed in the text) collapse to
one record, preferring an approved one. (3) Cross-system: each distinct tool line is matched to an unclaimed ERP
line, one-to-one. Pass 1 uses PO reference (normalised from `PO#4510040036` / `4510005953/10.0`) plus net amount
within 2 cents, with ties broken by PO line, quantity and description similarity. When the stated tax rate
fails, the 0% / default rate is tried, but then quantity or text must agree. Pass 2, for tool lines with no PO,
uses supplier plus net amount plus quantity/text. It is guarded by an id-sequence model learned from pass 1
(record ids in both systems grow with time; correlation 0.9997), which rejects same-supplier, same-amount
coincidences years apart. Matched tool lines are DUPLICATE; what remains is tool-only spend.

**De-duplication.** Suppliers: name key (accents, case, punctuation, spacing and legal forms such as PTY/LTD/CC/GmbH
removed; OCR look-alikes folded), linked to vendor ids except catch-all ids used for many names. Typing variants are
linked by character n-gram similarity, verified by edit distance (≤1–2 edits, never across two master-data names, and
never if a digit changes). Items: item code ↔ normalised description, except "dummy" codes carrying many different
texts. Fuzzy links follow the same rules, and two descriptions whose multi-digit numbers differ (1500MM vs 1800MM)
never link. All links go into union-find. When labels hold the client's golden keys (A), they guide the grouping
and new records join through name or code.

**Exclusions.** Status → CANCELLED / NOT_APPROVED through a generic vocabulary (unknown values are reported).
NON_PROCUREMENT comes from built-in multilingual patterns on account names and descriptions: intercompany, tax
accounts and tax-only lines, royalties, partner billing, financing, revenue, payroll, accounting entries. Services
*about* these (tax advisory) are kept, and a rule switches itself off if the company's tree has a category for that
topic. Config rules can add more, and exclusions the analyst made consistently (≥3 times, never included) are
generalised.

**Classifying into an unseen tree.** (1) A lexical scorer builds a vocabulary per leaf from the tree: its own and
ancestors' names (weighted by level), compounds, acronyms ("Can Body Stock" → CBS), and a generic procurement
lexicon of 72 concepts in several languages. Each concept attaches to whichever categories contain one of its
trigger words, so no category name appears in code. Line tokens are expanded for SAP-style abbreviations
(CNVY → conveyor, STNEDST → "stone dust") by consonant skeleton and prefix. (2) Self-training: confident,
unambiguous lexical matches become pseudo-labels for a linear model over word and char n-grams plus supplier,
account, item and context features. It carries evidence to lines whose own text says nothing. Final scores blend
both by lexical strength. With ≥200 labels (A, B), the model trains on the labels instead. The optional LLM labels
the highest-spend lines first.

**Measured accuracy** (by count / by spend; scripts in the repo):

| Test | L1 | L2 | L3 |
|---|---|---|---|
| A, labels hidden, tree only (37k ERP lines) | 0.61 / 0.76 | 0.37 / 0.62 | 0.32 / 0.55 |
| B, labels hidden, tree only (vs 2,550 seed labels) | 0.88 / 0.93 | 0.83 / 0.84 | 0.52 / 0.34 |
| A2 simulation: last 30% of A by id, trained on first 70% | 0.94 / 0.94 | 0.87 / 0.89 | 0.86 / 0.87 |
| B, trained on 70% of seed labels, held-out 30% | 0.97 / 0.98 | 0.96 / 0.98 | 0.95 / 0.98 |
| A label-trained classifier, 5-fold CV | | | 0.98 |

A's tree is narrow and mining-specific (the same spare part goes to "CM", "SC" or "RB" maintenance spares by
machine), which is why tree-only is hardest there. B's generic tree is the better proxy for an unseen company.
De-duplication on the A2 simulation: supplier pairwise F1 0.98, item F1 0.95. Unguided (no labels), supplier F1 is
1.00 outside the client's catch-all "other suppliers" key, which no method can infer without labels. With labels
hidden, B's exclusion rules reproduce the analyst's 450 exclusions with 0 false positives on the 2,550 included lines.

**Runtime.** A (65,549 records) about 20 s; B (41,028) about 15 s, end to end on one CPU, so 25,000 lines take
well under a minute.

**LLM cost.** None by default. With `--llm`: about 1.1–1.4k input and 0.1k output tokens per 40-line call with a
40-leaf tree, i.e. about 35k input and 3k output tokens per 1,000 lines (about twice that for a 150-leaf
tree), capped at `max_items` distinct lines per run. Multiply by the per-token price of the chosen model.
