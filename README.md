# spendcube — Cantill.ai Data & ML task

A config-driven tool that turns a company's raw purchase data and its category tree into a spend cube:
every record counted once, net of tax, in the reporting currency, with a reason code for everything
excluded, de-duplicated supplier and item keys, and a category (to the tree's lowest level) with a confidence.

* `APPROACH.md`: one-page design, measured accuracy, runtime, LLM cost
* `DECISIONS.md`: what I found in the data and what I decided
* `outputs/`: A and B outputs at the build deadline
* `submission/`: left empty until the blind test; then only the C config and the A2, B2 and C outputs

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

Put the kit's data next to the code (the folder is git-ignored, so **no task data is ever committed**):

```
data/company_A/  erp_goods_receipts.csv  invoice_tool_export.csv  labels_A.csv  taxonomy_A.csv
data/company_B/  invoice_lines.csv  fx_rates_usd.csv  seed_labels.csv  taxonomy_B.csv
```

## The one command

```bash
python run.py --config configs/A.yaml --data data/company_A --out outputs/<yourname>__A.csv
python run.py --config configs/B.yaml --data data/company_B --out outputs/<yourname>__B.csv
```

Without `--out`, the file goes to `submission/<yourname>__<company>.csv` (`--name`, `--suffix`), the layout the blind
test asks for. Each run also writes `audit/<file>_audit.csv`,
which explains every record: which record it duplicates and how it was matched, which rule excluded it, the
raw/cleaned amount, tax rate and FX rate used, the parsed date, data-quality flags, how it was classified and the
second-choice category. A and B run in about 20 s and 15 s (65k and 41k records).

Options: `--no-labels` (classify from the tree alone), `--reference <dir>` (earlier labelled data of the same
company), `--suffix A2` (output name), `--llm` (optional, see below), `-q` (quiet).

## Blind test (Thu 19 Nov)

New A and B records use the existing configs. The original kit folders stay in `data/`: they are the
`reference_dir` of each config, so their labels train the classifier and guide supplier/item keys for the new records.

```bash
python run.py --config configs/A.yaml --data <A2 folder> --name <yourname> --suffix A2
python run.py --config configs/B.yaml --data <B2 folder> --name <yourname> --suffix B2
```

Company C: draft a config from its files, check it line by line, then run.

```bash
python suggest_config.py --file <C folder>/<lines>.csv --taxonomy <C folder>/<tree>.csv --company C --currency USD > submission/C.yaml
python run.py --config submission/C.yaml --data <C folder> --name <yourname>
```

If a delivered file has a different name from the one in the config, the tool picks the file in the folder whose
header holds the configured columns, and says so in the log. The same applies to the tree and the FX table, so
A2 and B2 run on the existing configs.

A company config describes the files, how columns map to amount, tax rate, currency, FX rate, date, supplier,
description, item code, PO, invoice number, status and account; how sources match; and extra rules. A new company
needs a new config, not new code. See `configs/A.yaml` (two systems, gross tool amounts) and `configs/B.yaml`
(many currencies, FX table, account-based exclusions) for every option in use.

## Optional LLM

Off by default; the tool is complete without it. With `--llm` and `ANTHROPIC_API_KEY` set, the highest-spend
distinct lines (up to `llm.max_items`, default 1,500) are labelled by the model in `llm.model` (default
`claude-haiku-5-5`). Those labels then train the classifier like analyst labels. With no key, it logs a warning
and runs without it. Cost is in `APPROACH.md`.

## Measuring

```bash
python evaluate.py --config configs/A.yaml --data data/company_A --labels data/company_A/labels_A.csv
python -m pytest -q tests
```

`evaluate.py` hides A's labels, runs the pipeline from the tree alone, and reports L1–L3 accuracy (by count and
by spend) plus supplier/item de-duplication precision, recall, F1 and ARI against the golden keys. It also
reports a 5-fold cross-validated accuracy of the label-trained classifier.

## Layout

```
run.py                 the one command
suggest_config.py      drafts a config for a new company
evaluate.py            accuracy against A's labels
configs/               one YAML per company
spendcube/
  ingest.py            read files by config, canonical columns, row hashes, flags
  parsing.py           numbers (1,234.56 / 1.234,56 / (12) / 12-), tax rates (15 / 0.15 / 15%), mixed date formats
  pipeline.py          tax, FX, exclusions, output and audit
  match.py             repeated exports, byte-identical rows, cross-system matching
  dedupe.py            supplier and item keys (normalise, link, fuzzy, union-find)
  exclusions.py        status vocabulary, non-procurement rules, rules learned from labels
  classify.py          tree-driven lexical scorer + self-training / label-trained model
  llm.py               optional LLM labeller
  resources/           generic status words, non-procurement patterns, procurement lexicon (no category names)
tests/                 unit tests (no task data needed)
```

## AI use

I built this with Claude (Anthropic) as a coding assistant. The tool itself makes no API calls unless `--llm` is
given.
