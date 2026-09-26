"""Download the public regulations that make up the "legal source of truth" part of the corpus.

Sources (all public domain / freely published government text):
  * eCFR versioner API: federal regulations as of a pinned snapshot date, so results are
    reproducible even when the regulation is amended later.
  * California Legislative Information: Gov. Code 12950.1 (harassment prevention training).

Each document is saved untouched to data/raw/ and recorded in data/raw/manifest.json with its
source URL, retrieval time and SHA-256, which is what incremental re-indexing keys on.

Usage:
    python data/fetch_regulations.py                  # default snapshot date
    python data/fetch_regulations.py --date 2026-09-24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent / "raw"
DEFAULT_DATE = "2026-09-01"
ECFR_URL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml?part={part}&section={section}"
USER_AGENT = "learning-compliance-rag/0.1 (research; contact via GitHub repo)"

# Sections chosen because they contain explicit *training* obligations (who, what, how often).
# That is the question space the assistant targets.
ECFR_SECTIONS: list[tuple[int, str, str]] = [
    (29, "1910", "1910.38"),    # Emergency action plans - employee training
    (29, "1910", "1910.95"),    # Occupational noise - annual training program
    (29, "1910", "1910.132"),   # PPE - general training / retraining
    (29, "1910", "1910.134"),   # Respiratory protection - annual training, fit testing
    (29, "1910", "1910.146"),   # Permit-required confined spaces - training
    (29, "1910", "1910.147"),   # Lockout/tagout - training and retraining
    (29, "1910", "1910.157"),   # Portable fire extinguishers - annual training
    (29, "1910", "1910.178"),   # Powered industrial trucks - operator training, 3-yr evaluation
    (29, "1910", "1910.1030"),  # Bloodborne pathogens - annual training
    (29, "1910", "1910.1200"),  # Hazard communication - employee information and training
    (45, "164", "164.308"),     # HIPAA Security Rule - security awareness training
    (45, "164", "164.530"),     # HIPAA Privacy Rule - workforce training
]

STATE_SECTIONS = [
    {
        "doc_id": "CA-GOV-12950.1",
        "citation": "Cal. Gov. Code § 12950.1",
        "url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=GOV&sectionNum=12950.1",
    }
]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return s


def _get(session: requests.Session, url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            # eCFR omits the charset header; requests would then fall back to latin-1 and
            # mangle "§" and em dashes. Both sources are UTF-8.
            if "charset" not in r.headers.get("Content-Type", "").lower():
                r.encoding = "utf-8"
            return r.text
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _extract_ca_section(html_text: str) -> str:
    """Keep only the statute body; the page wraps it in navigation and scripts."""
    start = html_text.find('id="codeLawSectionNoHead"')
    if start < 0:
        raise ValueError("statute body not found; page layout may have changed")
    return html_text[start - 5 : start + 20000]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=DEFAULT_DATE, help="eCFR point-in-time date (YYYY-MM-DD)")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = _session()
    manifest: list[dict] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for title, part, section in ECFR_SECTIONS:
        url = ECFR_URL.format(date=args.date, title=title, part=part, section=section)
        text = _get(session, url)
        doc_id = f"{title}-CFR-{section}"
        path = RAW_DIR / f"{doc_id}.xml"
        path.write_text(text, encoding="utf-8")
        manifest.append(
            {
                "doc_id": doc_id,
                "citation": f"{title} CFR {section}",
                "format": "ecfr_xml",
                "path": path.name,
                "source_url": url,
                "as_of": args.date,
                "retrieved_at": now,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
        print(f"fetched {doc_id:22s} {len(text):>8,d} chars")

    for spec in STATE_SECTIONS:
        body = _extract_ca_section(_get(session, spec["url"]))
        path = RAW_DIR / f"{spec['doc_id']}.html"
        path.write_text(body, encoding="utf-8")
        manifest.append(
            {
                "doc_id": spec["doc_id"],
                "citation": spec["citation"],
                "format": "ca_leginfo_html",
                "path": path.name,
                "source_url": spec["url"],
                "as_of": re.sub(r"T.*", "", now),
                "retrieved_at": now,
                "sha256": hashlib.sha256(body.encode()).hexdigest(),
            }
        )
        print(f"fetched {spec['doc_id']:22s} {len(body):>8,d} chars")

    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest with {len(manifest)} documents -> {RAW_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
