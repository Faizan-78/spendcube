"""Robust parsing of amounts, rates and dates. Nothing here is company-specific."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_BLANK = {"", "nan", "none", "null", "n/a", "na", "-", "--", "#n/a"}


def is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().str.lower().isin(_BLANK)


def _decimal_vote(values: pd.Series) -> str:
    """Decide whether ',' or '.' is the decimal separator for a column."""
    s = values.dropna().astype(str)
    both = s[s.str.contains(",", regex=False) & s.str.contains(".", regex=False)]
    if len(both):
        comma_last = (both.str.rfind(",") > both.str.rfind(".")).mean()
        return "," if comma_last > 0.5 else "."
    comma_dec = s.str.contains(r",\d{1,2}$").sum() + s.str.contains(r",\d{4,}$").sum()
    dot_dec = s.str.contains(r"\.\d{1,2}$").sum() + s.str.contains(r"\.\d{4,}$").sum()
    return "," if comma_dec > dot_dec else "."


def parse_number(series: pd.Series, decimal: str = "auto") -> pd.Series:
    """Parse numbers written as 1,234.56 / 1.234,56 / (123.45) / 123.45- / 'R 1 234'.

    Unparseable values become NaN (the caller flags them).
    """
    s = series.astype("string").str.strip()
    s = s.mask(is_blank(s))
    neg = s.str.contains(r"^\(.*\)$|-$|^-", na=False)
    s = s.str.replace(r"[^\d,.\-eE+]", "", regex=True).str.replace(r"^[+-]|-$", "", regex=True)
    dec = _decimal_vote(s) if decimal == "auto" else decimal
    has_both = s.str.contains(",", regex=False, na=False) & s.str.contains(".", regex=False, na=False)
    comma_dec = pd.Series(dec == ",", index=s.index)
    # a value holding both separators decides for itself: the last one is the decimal mark
    comma_dec = comma_dec.where(~has_both, s.str.rfind(",") > s.str.rfind("."))
    as_comma = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    as_dot = s.str.replace(",", "", regex=False)
    s = as_dot.where(~comma_dec.fillna(False).astype(bool), as_comma)
    out = pd.to_numeric(s, errors="coerce")
    return out.where(~neg, -out).astype("float64")


def parse_rate(series: pd.Series, default: float | None = None) -> pd.Series:
    """Tax rate written as 15, 0.15 or 15% -> 0.15. Blank -> default."""
    r = parse_number(series.astype("string").str.replace("%", "", regex=False))
    r = r.where(r.abs() <= 1, r / 100.0)
    if default is not None:
        r = r.fillna(default)
    return r


_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
    "%d-%b-%y", "%d-%b-%Y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%d.%m.%Y", "%d.%m.%y",
]
_DAY_FIRST = ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"]
_MONTH_FIRST = ["%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y"]


def parse_date(series: pd.Series, formats: list[str] | None = None, dayfirst="auto"):
    """Parse a column holding a mix of date formats.

    Returns (dates, flag) where flag is '', 'DATE_MISSING', 'DATE_UNPARSEABLE' or
    'DATE_IMPLAUSIBLE' (more than 3 years away from the column's median year).
    """
    raw = series.astype("string").str.strip()
    blank = is_blank(raw)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    def attempt(fmt_list):
        nonlocal out
        for f in fmt_list:
            todo = out.isna() & ~blank
            if not todo.any():
                return
            out.loc[todo] = pd.to_datetime(raw[todo], format=f, errors="coerce")

    attempt(list(formats or []) + _FORMATS)
    todo = out.isna() & ~blank
    if todo.any():
        if dayfirst == "auto":
            d = pd.to_datetime(raw[todo], format=_DAY_FIRST[0], errors="coerce").notna().sum()
            m = pd.to_datetime(raw[todo], format=_MONTH_FIRST[0], errors="coerce").notna().sum()
            dayfirst = d >= m
        attempt(_DAY_FIRST + _MONTH_FIRST if dayfirst else _MONTH_FIRST + _DAY_FIRST)
    todo = out.isna() & ~blank
    if todo.any():  # Excel serial numbers
        n = pd.to_numeric(raw[todo], errors="coerce")
        ok = n.between(20000, 80000)
        out.loc[n[ok].index] = pd.to_datetime("1899-12-30") + pd.to_timedelta(n[ok], unit="D")
    flag = pd.Series("", index=series.index, dtype="object")
    flag[blank] = "DATE_MISSING"
    flag[out.isna() & ~blank] = "DATE_UNPARSEABLE"
    years = out.dt.year
    if years.notna().any():
        med = years.median()
        bad = out.notna() & ((years - med).abs() > 3)
        flag[bad] = "DATE_IMPLAUSIBLE"
    return out, flag


def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((b == 0) | pd.isna(b), np.nan, a / b)
