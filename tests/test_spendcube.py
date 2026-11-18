"""Unit tests that need no task data:  python -m pytest -q"""
import json

import numpy as np
import pandas as pd

from spendcube.dedupe import number_signature, supplier_name_key, supplier_keys
from spendcube.exclusions import status_class
from spendcube.ingest import load_status_map
from spendcube.llm import AnthropicLabeller, parse_answers
from spendcube.parsing import parse_date, parse_number, parse_rate
from spendcube.textnorm import levenshtein, text_key


def test_numbers():
    s = pd.Series(["5,266.63", "8721.47", "(1,000.00)", "1.234,56", "12-", "abc", None])
    out = parse_number(s).tolist()
    assert out[:5] == [5266.63, 8721.47, -1000.0, 1234.56, -12.0]
    assert np.isnan(out[5]) and np.isnan(out[6])


def test_rates():
    assert parse_rate(pd.Series(["15", "0.15", "15%", None, "0"]), 0.15).tolist() == [0.15, 0.15, 0.15, 0.15, 0.0]


def test_dates():
    d, f = parse_date(pd.Series(["2021-08-16", "27-Mar-24", "31/12/2022", None, "garbage"]))
    assert d.iloc[1] == pd.Timestamp("2024-03-27") and d.iloc[2] == pd.Timestamp("2022-12-31")
    assert f.tolist()[3:] == ["DATE_MISSING", "DATE_UNPARSEABLE"]


def test_supplier_names():
    assert supplier_name_key("YARAX  TECHNOLOGIES (PTY) LTD") == supplier_name_key("Yarax Technologies (Pty) Ltd")
    df = pd.DataFrame({"supplier_name": ["KINORA INDUSTRIAL SUPPLIES ENGINEERING", "KINORA  INDUSTRIAL SUPPLIES ENGINEERNG", "OTHER CO"],
                       "supplier_id": ["", "", ""]})
    k = supplier_keys(df)
    assert k[0] == k[1] != k[2]


def test_numbers_block_item_links():
    assert number_signature("ROD ANCHR;DIA:20MM,1800MM") != number_signature("ROD ANCHR;DIA:20MM,1500MM")
    assert number_signature("R0D ANCHR;DIA:20MM,1800MM") == number_signature("ROD ANCHR;D1A:20MM,1800MM")
    assert text_key("P0WDER;STNEDST") == text_key("POWDER;STNEDST")
    assert levenshtein("kitten", "sitting", 5) == 3


def test_status():
    m = load_status_map({})
    s = status_class(pd.Series(["Paid", "Approved", "Cancelled", "On Hold", "Rejected", ""]), m).tolist()
    assert s == ["INCLUDE", "INCLUDE", "CANCELLED", "NOT_APPROVED", "NOT_APPROVED", "INCLUDE"]


def test_llm_parse_and_call(monkeypatch):
    assert parse_answers('noise {"answers": [0, 2, -1]}', 3, 3) == [0, 2, None]
    assert parse_answers("garbage", 2, 3) == [None, None]

    class Tree:
        leaves = [("A", "a1"), ("B", "b1")]
        def __len__(self):
            return 2

    lab = AnthropicLabeller("m", "k")
    monkeypatch.setattr(lab, "_call", lambda prompt: json.dumps({"answers": [1, 0]}))
    assert lab.label([("valve", ""), ("freight", "")], Tree()) == [1, 0]
