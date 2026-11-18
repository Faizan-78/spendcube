"""Optional LLM labelling of the highest-spend distinct lines (off by default).

The tool runs fully without it. When switched on (run.py --llm) and a key is present, the LLM labels up to
`max_items` distinct lines, highest spend first; those labels then train the classifier like analyst labels,
so the rest of the file benefits too. Without a key it logs a warning and the run continues unchanged.

Config (all optional):
    llm:
      provider: anthropic           # Messages API over HTTPS, no SDK needed
      model: claude-haiku-5-5
      api_key_env: ANTHROPIC_API_KEY
      max_items: 1500               # distinct lines sent, highest spend first
      batch_size: 40
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request

log = logging.getLogger("spendcube")

PROMPT = """You classify purchase lines into a company's procurement category tree.

Categories (number: path):
{tree}

For each line below, choose the single best category number. Use the description, the account and any
context. If a line gives no clue at all, answer -1. Reply with JSON only: {{"answers": [n1, n2, ...]}}
with exactly {k} integers in the order of the lines.

Lines:
{lines}"""


class AnthropicLabeller:
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, model, key, max_items=1500, batch_size=40, timeout=60):
        self.model, self.key = model, key
        self.max_items, self.batch_size, self.timeout = max_items, batch_size, timeout
        self.tokens_in = self.tokens_out = 0

    def _call(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "max_tokens": 1024, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(self.url, data=body, method="POST", headers={
            "x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                u = data.get("usage", {})
                self.tokens_in += u.get("input_tokens", 0)
                self.tokens_out += u.get("output_tokens", 0)
                return "".join(c.get("text", "") for c in data.get("content", []))
            except Exception as e:  # rate limits / transient errors: back off, then give up on this batch
                log.warning("LLM call failed (%s), attempt %d", e, attempt + 1)
                time.sleep(2 ** attempt)
        return ""

    def label(self, items, tree):
        """items: list of (description, context). Returns a leaf index or None per item."""
        tree_txt = "\n".join(f"{i}: " + " > ".join(x for x in leaf if x) for i, leaf in enumerate(tree.leaves))
        out = []
        for s in range(0, len(items), self.batch_size):
            chunk = items[s:s + self.batch_size]
            lines = "\n".join(f"{j + 1}. {d[:200]!s} || {c[:120]!s}" for j, (d, c) in enumerate(chunk))
            txt = self._call(PROMPT.format(tree=tree_txt, k=len(chunk), lines=lines))
            out += parse_answers(txt, len(chunk), len(tree))
        log.info("LLM usage: %d input tokens, %d output tokens for %d lines", self.tokens_in, self.tokens_out, len(items))
        return out


def parse_answers(txt: str, k: int, n_leaves: int):
    try:
        m = re.search(r"\{.*\}", txt, re.S)
        ans = json.loads(m.group(0))["answers"] if m else []
    except (ValueError, KeyError, TypeError):
        ans = []
    if len(ans) != k:
        return [None] * k
    return [int(a) if isinstance(a, (int, float)) and 0 <= int(a) < n_leaves else None for a in ans]


def make_llm(cfg: dict):
    c = cfg.get("llm") or {}
    key = os.environ.get(c.get("api_key_env", "ANTHROPIC_API_KEY"), "")
    if not key:
        log.warning("--llm given but no API key in $%s: running without the LLM", c.get("api_key_env", "ANTHROPIC_API_KEY"))
        return None
    if c.get("provider", "anthropic") != "anthropic":
        raise ValueError("only provider: anthropic is implemented")
    return AnthropicLabeller(c.get("model", "claude-haiku-5-5"), key, c.get("max_items", 1500), c.get("batch_size", 40))
