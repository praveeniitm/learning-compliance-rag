"""Offline unit tests (no API calls): tokenizer, ACL / as-of filter, chunking, gold coverage, PII rail."""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "sk-test-offline")
os.environ["LLM_CACHE"] = "0"
sys.path.insert(0, str(Path(__file__).parents[1] / "eval"))

from lcrag.corpus import load_documents, split_documents  # noqa: E402
from lcrag.store import allowed, tokenize  # noqa: E402

DOCS = load_documents()
CHUNKS = split_documents(DOCS)


def test_tokenizer_keeps_citations_and_codes():
    toks = tokenize("See 29 CFR 1910.178(l)(4)(iii) and course EHS-FL-201 for near-miss events")
    assert {"1910.178", "1910.178(l)", "1910.178(l)(4)", "1910.178(l)(4)(iii)", "ehs-fl-201"} <= set(toks)
    assert "near" in toks and "miss" in toks  # ordinary hyphenated words still split


def test_acl_filter():
    audit = {"access": "compliance,ehs,lms_admin"}
    assert allowed(audit, "ehs") and allowed(audit, "compliance")
    assert not allowed(audit, "employee") and not allowed(audit, "manager")
    assert allowed({"access": "all"}, "employee") and allowed({}, "employee")


def test_as_of_filter():
    v31 = {"effective_date": "2024-07-01", "superseded_on": "2026-01-01"}
    v32 = {"effective_date": "2026-01-01"}
    assert allowed(v31, as_of="2025-06-01") and not allowed(v32, as_of="2025-06-01")
    assert not allowed(v31, as_of="2026-03-01") and allowed(v32, as_of="2026-03-01")


def test_every_document_has_access_metadata_or_is_public_law():
    for d in DOCS:
        assert d.metadata["doc_type"] in ("regulation", "statute") or "access" in d.metadata, d.metadata["doc_id"]


def test_chunks_carry_contextual_header_and_paragraph_cites():
    c = next(c for c in CHUNKS if "[29 CFR 1910.178(l)(4)(iii)]" in c.page_content)
    assert c.page_content.startswith("[29 CFR 1910.178]") and "forklift" in c.page_content.splitlines()[0]
    sup = [c for c in CHUNKS if c.metadata["doc_id"] == "POL-CT-001-v3.1"]
    assert sup and all("SUPERSEDED" in c.page_content for c in sup)


def test_every_gold_passage_is_resolvable():
    from evaluate import QA, covered

    missing = [(q["id"], g) for q in QA for g in q["gold"] if not any(covered(c, [g]) for c in CHUNKS)]
    assert not missing


def test_pii_regex_rail_blocks_without_llm_call():
    from lcrag.guardrails import check_input

    assert check_input("My SSN is 123-45-6789, which courses do I need?") == "regex check input"
