#!/usr/bin/env python3
"""One command, end to end:

    python run.py --config configs/A.yaml --data data/company_A --name yourname

writes submission/<name>__<company>.csv, plus audit/<name>__<company>_audit.csv explaining every decision.
Blind-test examples are in README.md.
"""
import argparse
import logging
import sys
from pathlib import Path

from spendcube.ingest import load_config
from spendcube.pipeline import run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="company config (YAML)")
    ap.add_argument("--data", required=True, help="folder holding the files named in the config")
    ap.add_argument("--reference", help="folder with earlier, labelled data of the same company (overrides reference_dir in the config)")
    ap.add_argument("--name", default="yourname", help="your name, used in the output file name")
    ap.add_argument("--suffix", help="output suffix; default = company from the config (use A2 / B2 at the blind test)")
    ap.add_argument("--out", help="full output path (overrides --name/--suffix)")
    ap.add_argument("--no-labels", action="store_true", help="ignore label files: classify from the tree alone")
    ap.add_argument("--llm", action="store_true", help="use the LLM configured under llm: in the config (needs an API key)")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(a.config)
    llm = None
    if a.llm:
        from spendcube.llm import make_llm
        llm = make_llm(cfg)
    out = a.out or str(Path("submission") / f"{a.name}__{a.suffix or cfg['company']}.csv")
    run(cfg, a.data, out, reference_dir=a.reference, use_labels=not a.no_labels, llm=llm, audit=not a.no_audit)


if __name__ == "__main__":
    sys.exit(main())
