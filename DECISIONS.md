# Decisions

What I found in the data, and what the tool does about it. Counts are from the kit files; per-record evidence
is in `audit/<name>__<company>_audit.csv`.

## Company A (ZAR)

### ERP goods receipts (system of record, 37,029 lines)
- `value_zar` is net and in ZAR for every line, including the 16 lines whose `document_currency` is EUR, CHF or MZN.
  It is used as is.
- 434 negative lines (returns/reversals) are kept as negative spend: the cube is net.
- 509 lines are worth exactly R0.01, apparently placeholder values on zero-value receipts. They are kept (they are
  real receipts, and the effect on totals is nil).
- `posting_date`, `vendor_id`, `uom` and `site_code` are blank on 66% of lines. Dates are not needed for A (no FX),
  so blank dates are only flagged `DATE_MISSING`.
- No line looks like non-procurement. A bare "JIB" (jib crane) once matched my partner-billing pattern (Joint
  Interest Billing). The pattern now needs the full phrase.

### Invoice tool (28,520 lines, gross)
- **Tax:** `vat_rate` is written as `15`, `0.15`, `15%`, `0` or blank. All are normalised to a fraction; blank → 15%
  (brief). The data confirms both rules: blank-rate lines match an ERP amount at ÷1.15 87% of the time (gross: 4%),
  and `0` lines match at the gross amount 85% of the time, so 0% is a real zero rate.
- **Amount format:** `5,266.63` and `5266.63` both occur; the parser handles thousand separators, decimal commas,
  `(123)` and `123-`.
- **Status:** Paid 19,837, Approved 8,180 → counted. Cancelled 188 → CANCELLED. On Hold 172 and Rejected 143 →
  NOT_APPROVED.
- **PO reference:** `4510040036`, `PO#4510040036`, `PO 4510040036`, `4510005953/10.0` (with PO line), or blank (18%).
  The tool extracts the 10-digit PO and the optional line.
- **Invoice numbers** come in five styles (`IN0914136`, `SI-2024-…`, `INV/2023/…`, `TAX INV …`, plain digits) and
  are not shared with the ERP, so they are used only inside the tool (repeated exports), never to match the ERP.
- **Dates:** 65% blank; the rest are ISO, `dd/mm/yyyy` or `dd-Mon-yy`. Day-first was chosen because values such as
  `31/12/2022` exist and no month > 12 appears in second position. Dates are flagged, not used for matching.
- **OCR noise:** `R0OF`, `B1T`, `P0WDER`, `5TONE`, dropped spaces and letters, `No Object` placeholder text.
  Comparisons fold 0/O, 1/I/L, 5/S, 8/B and ignore spacing.

### The same invoice exported twice
2,358 extra copies share supplier, invoice number, net amount and quantity, with differently OCR'd text and
sometimes different date formats or PO reference styles. **Decision:** one copy per group is kept (an approved one
when there is one; otherwise the first); the rest are DUPLICATE. Their status is never mixed within a group.

### Matching the two systems
- 61% of tool PO references exist in the ERP file; the rest are tool-only purchases.
- **Pass 1:** PO + net amount (±2 cents). Two lines on the same PO with the same amount are told apart by PO line,
  quantity and description similarity, and each ERP line is claimed once. 17,084 tool records.
- **Pass 2**, for tool lines with no PO reference: supplier + net amount + (same quantity or similar text). Two
  coincidences are possible here (same stone-dust delivery price every month), so a candidate must also sit at
  the right point of the id sequence: record ids in both files grow with time (correlation 0.9997). The mapping
  is learned from pass-1 matches, and candidates further away than the 99.5th-percentile residual are rejected.
  3,384 tool records. With a PO reference that is *not* in the ERP file, no amount-only match is attempted.
- 877 tool lines carry a PO that is in the ERP, but no ERP line on it has their amount (different price,
  partial delivery). **Decision:** they are treated as tool-only spend rather than forced onto an ERP line.
  This may slightly over-count, but forcing them would hide real invoice-vs-receipt differences.
- **Status vs duplicate:** the brief says a purchase in both systems is counted in the ERP and the tool copy is
  DUPLICATE. So cross-system DUPLICATE takes priority over the tool's status. Matching prefers approved tool lines.
- Result: 21,288 tool records DUPLICATE; 6,746 tool-only lines counted (R482m), on top of ERP spend of R2,553m.

### Suppliers
193 names in the ERP; the client's labels have 70 golden keys. One golden key (`SG-f85711`, 20,891 lines, 122
names, almost all without a vendor id) is a catch-all "other suppliers" bucket. Outside it, names map 1:1 to golden
keys after normalisation, and my unguided keys reproduce them exactly (pairwise F1 1.00). **Decision:** the golden
keys in the labels guide grouping (new records join through their name), so the catch-all is reproduced for A2.
Without labels, every real name stays its own supplier.

### Items
- `material_number` → golden item is 1:1, except `M1000032`, a dummy code used for free-text lines with hundreds of
  different descriptions. Codes that carry many unrelated texts are treated as dummies and their lines are keyed by
  description.
- Repairs of the same unit are booked under many material numbers with one text (`REPR;FAN ELEC;22 KW…`: 155
  codes, one golden item), so codes sharing a description are merged.
- The golden item ignores `service_short_text` (warranty numbers, months), so the item key uses
  `material_short_text` only. Service text still feeds classification.
- Lines that differ only in a multi-digit number (1500MM vs 1800MM anchor) are never merged, even through a chain
  of OCR variants.
- ERP lines with no material number have no golden item; their key comes from the description.

### Classification
Labels cover every ERP line, so A is classified by a model trained on them (5-fold CV 0.98 at L3). Tool-only lines
borrow their text, supplier and item. With labels hidden, see `APPROACH.md`.

## Company B (USD)

### Records
- 41,028 lines from 8 ERPs. There is no status column. **Decision:** a line counts unless a rule excludes it.
  `paid_date` is blank on 10,322 lines, but the analyst included 617 of the 655 such seed lines, so a blank paid
  date is not read as "not approved".
- `line_description` is blank or missing on 40% of lines (ERP-B6 100%, B1/B8 96%). Those lines are classified from
  supplier, account, item and context columns.
- 3 lines in currency `RCN` have no amount. They stay included with spend 0 and are flagged `AMOUNT_MISSING`.

### What is not spend (all 450 analyst exclusions are NON_PROCUREMENT)
- **Intercompany:** account descriptions `IC Payables`, `I/C Payable`, `Intercompany - CZ`, and JDE ledger codes
  `IC05` … `IC70` (an IC prefix plus a number; added as a config rule because that code is B-specific). 2,041 lines.
- **Taxes booked as lines:** `State & Local Taxes -`, `Sales Tax Expense -`, `Sales Tax -`, `GST Input Tax -`. 447 lines.
  Tax *services* (`Tax Services and Garnishments`, `PayTech Tax Consultant`, `Blanket PO for Tax Work`) and customs
  duty stay in (B's tree has a customs category).
- With labels hidden, these rules reproduce all 450 analyst exclusions and exclude none of the 2,550 lines the
  analyst included.
- Negative lines (credit notes, 1–2% of lines) are spend, counted net, as in the seed labels.

### Duplicates
613 lines are identical to another line in every column except `line_id` (890 of them in ERP-B3), e.g. the same
invoice line number twice with the same amount. **Decision:** keep one, the rest DUPLICATE (604; 9 more were
already NON_PROCUREMENT). Reversal pairs (+x / −x on the same invoice) are not duplicates; they net to zero.

### Currency
- USD lines are used as is, even the 1,279 USD lines from the Mexican JDE ledgers that carry an MXN rate (~17.7).
- `currency_rate` (units per USD) is used when present and within 3× of the monthly table. 60 lines in ERP-B3
  (EUR/CHF) and ERP-B8 (ARS) carry a rate that is clearly another currency's (EUR at 18.6 ≈ the MXN rate).
  **Decision:** the table rate is used and the line is flagged `FX_RATE_IMPLAUSIBLE`. The amounts look like
  EUR/CHF prices, not MXN.
- No rate → table rate for the invoice month. When the invoice date is implausible (2018, 2027; flagged
  `DATE_IMPLAUSIBLE`) the posting month is used. 5 lines have no table entry for their month → nearest month.

### Suppliers and items
- Only vendor ids, no names: one supplier per vendor id. The same company under different ids in different ERPs
  cannot be recognised without names.
- `MAT-1000007` (10,484 lines) is a dummy code for non-catalogue purchases; those lines are keyed by description.
  Lines with neither a usable code nor a description get `<code> (no description)` or `unspecified`.

### Classification
2,550 seed labels train the model (held-out accuracy 0.95 at L3, 0.98 by spend). The seed labels are described as
"useful, not perfect". A labelled line keeps its label, since the analyst saw it; every other line gets the
model's prediction, so a one-off analyst slip doesn't spread to lines it doesn't share text with.

## General decisions
- **Every input record is in the output**, excluded ones too, with spend 0 and a reason code.
- **Priority of exclusion reasons:** cross-system DUPLICATE → CANCELLED → NOT_APPROVED → NON_PROCUREMENT →
  repeated-export / identical-row DUPLICATE.
- **Unparseable values are flagged, never silently dropped:** `AMOUNT_UNPARSEABLE`, `AMOUNT_MISSING`, `DATE_MISSING`,
  `DATE_UNPARSEABLE`, `DATE_IMPLAUSIBLE` (more than 3 years from the file's median year), `FX_MISSING`,
  `FX_NEAREST_MONTH`, `FX_RATE_IMPLAUSIBLE`.
- **Unknown status values** are reported and treated as NOT_APPROVED (confidence 0.6) until mapped in the config:
  safer to under-count than to count rejected invoices.
- **Confidence:** for excluded records, how certain the exclusion is (status 0.99, PO match 0.97, supplier+amount
  match 0.85, pattern rule 0.9). For included records, the classifier's calibrated score (with labels hidden on B,
  lines scored above 0.8 are right 82% of the time; below 0.2, 4%).
- **Reference data:** at the blind test the kit folders are the reference. Their labels apply only to their own
  records, never to a new record that reuses an id.
