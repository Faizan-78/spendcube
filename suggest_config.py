#!/usr/bin/env python3
"""Draft a company config from a new invoice-line file (for the blind test's company C).

    python suggest_config.py --file data/company_C/<lines>.csv --taxonomy data/company_C/<tree>.csv \
        --company C --currency USD > configs/C.yaml

It guesses each field from column names and contents and prints a YAML with the evidence as comments.
Always read it through; it is a starting point, not a decision.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from spendcube.ingest import read_csv
from spendcube.parsing import parse_date, parse_number

HINTS = {
    "id": [r"^(line|record|row|txn|transaction)_?id$", r"^id$", r"line.?id", r"record.?id"],
    "amount": [r"net.?amount", r"line.?amount", r"amount.?(usd|net|local|doc)", r"^amount$", r"value", r"spend", r"net", r"total", r"amount"],
    "tax_rate": [r"(vat|tax|gst).?(rate|pct|percent|%)"],
    "currency": [r"^curr(ency)?(_code)?$", r"currency", r"\bccy\b|^cur$"],
    "fx_rate": [r"(fx|exchange|currency).?rate", r"\brate\b"],
    "date": [r"invoice.?date", r"posting.?date", r"doc(ument)?.?date", r"txn.?date", r"date"],
    "supplier_name": [r"supplier.?name", r"vendor.?name", r"supplier$", r"vendor$", r"payee", r"merchant"],
    "supplier_id": [r"(supplier|vendor).?(id|no|number|code|num)"],
    "description": [r"(line|item|invoice)?.?desc", r"description", r"line.?text", r"short.?text", r"item.?name", r"memo", r"narrative", r"text"],
    "item_code": [r"(material|item|product|part|sku).?(no|number|code|id|num)?$"],
    "quantity": [r"^qty$|quantity"],
    "po": [r"^po$|po.?(number|no|num|ref|reference)|purchase.?order"],
    "invoice_number": [r"invoice.?(number|no|num|id)|^inv.?(no|num)"],
    "status": [r"status|state|approval"],
    "gl_account": [r"(gl|ledger|account).?(account|code|no|number)?$"],
    "gl_description": [r"(gl|account|ledger).?(desc|name|description)"],
}


def pick(cols, field, used):
    for pat in HINTS[field]:
        for c in cols:
            if c not in used and re.search(pat, c, re.I):
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--company", default="C")
    ap.add_argument("--currency", default="USD")
    a = ap.parse_args()
    df = read_csv(Path(a.file))
    cols, used, f = list(df.columns), set(), {}
    for field in ["id", "gl_description", "gl_account", "supplier_id", "supplier_name", "invoice_number", "po", "item_code",
                  "tax_rate", "fx_rate", "currency", "quantity", "status", "date", "description", "amount"]:
        c = pick(cols, field, used)
        if c:
            f[field] = c
            used.add(c)
    tree = pd.read_csv(a.taxonomy, dtype=str)
    out = [f"# drafted by suggest_config.py from {Path(a.file).name}: CHECK EVERY LINE", f"company: {a.company}",
           f"reporting_currency: {a.currency}", "", "taxonomy:", f"  file: {Path(a.taxonomy).name}",
           f"  levels: [{', '.join(tree.columns)}]", "", "sources:", "  - name: LINES", f"    file: {Path(a.file).name}",
           f"    id_column: {f.get('id', cols[0])}", "    fields:"]
    for k in ["amount", "tax_rate", "currency", "fx_rate", "date", "supplier_name", "supplier_id", "description", "item_code",
              "quantity", "po", "invoice_number", "status", "gl_account", "gl_description"]:
        if k in f:
            col = df[f[k]]
            ex = ", ".join(repr(v)[:30] for v in col.dropna().astype(str).unique()[:3])
            out.append(f"      {k}: {f[k] if k != 'description' else '[' + f[k] + ']'}   # e.g. {ex}")
    out.append("    amount: {basis: net}          # gross? -> {basis: gross, default_tax_rate: 0.xx} and set tax_rate")
    rest = [c for c in cols if c not in used]
    looks_date = lambda c: parse_date(df[c].dropna().head(200))[1].eq("").mean() > 0.8 if df[c].notna().any() else False
    cat = [c for c in rest if 1 < df[c].nunique() <= max(50, len(df) // 50) and not re.search(r"date", c, re.I) and not looks_date(c)]
    txt = [c for c in rest if df[c].astype(str).str.contains(r"[A-Za-z]{4}", na=False).mean() > 0.3 and c not in cat]
    out.append("    classify:")
    out.append(f"      text: [{', '.join(txt[:4])}]          # free-text context columns")
    out.append(f"      categorical: [{', '.join(cat[:8])}]  # low-cardinality context (types, plants, cost centres)")
    out += ["", "duplicates:", "  exact_rows: true", ""]
    if "status" in f:
        out.append(f"# status values: {sorted(df[f['status']].dropna().unique())[:15]}")
    if "date" in f:
        d, fl = parse_date(df[f["date"]])
        out.append(f"# date parse: {(fl == '').mean():.1%} ok, flags {fl[fl != ''].value_counts().to_dict()}")
    if "amount" in f:
        n = parse_number(df[f["amount"]])
        out.append(f"# amount parse: {n.notna().mean():.1%} ok, sum {n.sum():,.2f}")
    if "currency" in f:
        out.append(f"# currencies: {df[f['currency']].value_counts().head(10).to_dict()}")
    out.append(f"# unused columns: {rest}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
