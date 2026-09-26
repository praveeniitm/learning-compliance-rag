"""Post-generation grounding checks. They run locally, add no LLM cost, and each one is independent.

1. Citation validity: every source tag the model cites ([S1]...) must exist in the context it was
   given. A tag that does not exist is a fabricated citation.
2. Numeric grounding: every number in a claim (intervals, day counts, hours, headcounts) must appear
   in at least one of the claim's cited sources. In this domain the most damaging hallucinations
   are wrong numbers ("every 2 years" instead of 3), and exact matching catches them cheaply.
3. Entailment (NLI): a small cross-encoder NLI model (DeBERTa-v3-xsmall, about 70M parameters)
   scores each claim against sentence windows of its cited sources. This catches paraphrased but
   unsupported claims that the numeric check cannot see. xsmall was chosen over DeBERTa-v3-small,
   DeBERTa-v3-base (MNLI/FEVER/ANLI) and DeBERTa-v3-large on a probe set. The larger models were
   not more accurate here; they were stricter about vocabulary mismatches such as
   forklift vs. powered industrial truck, and 3-6x slower.

Why not an LLM judge at runtime: it doubles latency and cost for every query, and it shares failure
modes with the generator. The LLM judge is used offline in evaluation, where its agreement with
these runtime checks is also measured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .embeddings import torch_device

NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
ENTAIL_THRESHOLD = 0.5
CONTRA_THRESHOLD = 0.5

_WORD_NUMS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "twelve": "12",
              "fifteen": "15", "thirty": "30", "annual": "12", "annually": "12", "biennial": "24"}
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)*\b")


@lru_cache(maxsize=1)
def _nli():
    from sentence_transformers import CrossEncoder

    m = CrossEncoder(NLI_MODEL, device=torch_device())
    labels = {lbl.lower()[:6]: i for i, lbl in m.config.id2label.items()}
    return m, labels["entail"], labels["contra"]


def _numbers(text: str) -> set[str]:
    """Numbers relevant to requirements; section/paragraph citations are excluded."""
    text = re.sub(r"\[S\d+\]", " ", text)
    text = re.sub(r"\b\d{2,5}\.\d+(\([^)]*\))*", " ", text)  # 1910.178(l)(4), 164.530
    text = re.sub(r"§\s*\d+(\.\d+)*", " ", text)  # policy clause refs like §7.1
    text = re.sub(r"\b[A-Z]{2,4}(-[A-Z0-9]+)+\b", " ", text)  # course / doc codes
    text = re.sub(r"\bv\d+(\.\d+)*\b", " ", text)
    return set(_NUM_RE.findall(text))


def _source_numbers(text: str) -> set[str]:
    nums = set(_NUM_RE.findall(text))
    low = text.lower()
    for w, n in _WORD_NUMS.items():
        if re.search(rf"\b{w}\b", low):
            nums.add(n)
    # "every 36 months" <-> "three years", "12 months" <-> "annually": accept unit conversions
    for n in list(nums):
        if n.isdigit():
            v = int(n)
            if v % 12 == 0 and v:
                nums.add(str(v // 12))
            nums.add(str(v * 12))
    return nums


@dataclass
class ClaimCheck:
    text: str
    sources: list[str]
    valid_citations: bool
    missing_numbers: list[str]
    entailment: float
    contradiction: float
    label: str  # "supported" | "contradicted" | "unverified"

    @property
    def supported(self) -> bool:
        return self.label == "supported"

    def to_dict(self) -> dict:
        return self.__dict__ | {"supported": self.supported}


@dataclass
class GroundingReport:
    """Three claim labels, because NLI "neutral" mixes two different cases:

    * contradicted: a cited source contradicts the claim, or a number in the claim appears in none
      of its sources. This is a hard failure and triggers revision.
    * unverified: neutral. This covers genuinely unsupported claims, but also correct claims that
      restate a detail from the question ("forklift operators *at Fresno* ...") that the source
      states only in general form. It is shown to the user as "not verified", not as an error.
    * supported: entailed by at least one window of a cited source, with every number grounded.
    """

    claims: list[ClaimCheck] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)

    @property
    def support_rate(self) -> float:
        return sum(c.supported for c in self.claims) / len(self.claims) if self.claims else 0.0

    @property
    def verdict(self) -> str:
        if not self.claims:
            return "no_claims"
        if self.invalid_citations:
            return "invalid_citation"
        if any(c.label == "contradicted" for c in self.claims):
            return "contradicted"
        if all(c.supported for c in self.claims):
            return "supported"
        return "partially_supported" if self.support_rate > 0 else "unsupported"

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "support_rate": round(self.support_rate, 3),
                "invalid_citations": self.invalid_citations, "claims": [c.to_dict() for c in self.claims]}


def _premise_windows(text: str, size: int = 2) -> list[str]:
    """Split a source into overlapping windows of `size` sentences (or table rows).

    Small NLI models are trained on single-sentence premises and lose accuracy on 300-token
    passages: a supported claim scored 0.07 against a whole policy section but above 0.9 against
    the sentence that states it. Scoring every window and taking the maximum (the SummaC approach)
    fixes that.
    """
    units = [u.strip() for u in re.split(r"(?<=[.:])\s+(?=[A-Z(\d])|\n", text) if u.strip()]
    if len(units) <= size:
        return [" ".join(units)] if units else [text]
    windows = [" ".join(units[i : i + size]) for i in range(len(units) - size + 1)]
    # The whole passage is scored as well, for claims that aggregate several sentences or rows
    # ("CLN-TEL is assigned EHS-EAP-100, HR-HAR-100 and ETH-COC-100").
    return windows + [text]


def _split_header(text: str) -> tuple[str, str]:
    """Sources are formatted as '<header>\n\n<body>' when the chunk has a contextual header."""
    if "\n\n" in text:
        head, body = text.split("\n\n", 1)
        if len(head) < 400:
            return head.replace("\n", " | "), body
    return "", text


def check_grounding(claims: list[dict], sources: dict[str, str], cited_in_answer: list[str]) -> GroundingReport:
    """`claims`: [{"text", "sources": ["S1", ...]}]; `sources`: tag -> source text."""
    report = GroundingReport(invalid_citations=sorted({s for s in cited_in_answer if s not in sources}))
    pairs, index = [], []
    for ci, c in enumerate(claims):
        for tag in c.get("sources", []):
            if tag in sources:
                header, body = _split_header(sources[tag])
                for window in _premise_windows(body):
                    # keep the source's title/heading trail so "(iii) ... every three years" still
                    # says it is about powered industrial truck operators
                    pairs.append((f"{header}\n{window}" if header else window, c["text"]))
                    index.append(ci)
    ent_p, con_p = [0.0] * len(claims), [0.0] * len(claims)
    if pairs:
        model, ent, con = _nli()
        scores = model.predict(pairs, apply_softmax=True, batch_size=16, show_progress_bar=False)
        for ci, row in zip(index, scores):
            ent_p[ci] = max(ent_p[ci], float(row[ent]))
            con_p[ci] = max(con_p[ci], float(row[con]))
    for ci, c in enumerate(claims):
        tags = c.get("sources", [])
        valid = bool(tags) and all(t in sources for t in tags)
        src_nums = set().union(*[_source_numbers(sources[t]) for t in tags if t in sources]) if valid else set()
        missing = sorted(n for n in _numbers(c["text"]) if n not in src_nums)
        if missing or (ent_p[ci] < ENTAIL_THRESHOLD and con_p[ci] >= CONTRA_THRESHOLD):
            label = "contradicted"
        elif valid and ent_p[ci] >= ENTAIL_THRESHOLD:
            label = "supported"
        else:
            label = "unverified"
        report.claims.append(ClaimCheck(c["text"], tags, valid, missing, round(ent_p[ci], 3), round(con_p[ci], 3),
                                        label))
    return report
