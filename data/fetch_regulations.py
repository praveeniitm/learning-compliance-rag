"""Download the public regulations in the corpus and save them as Markdown with front matter.

* Federal: eCFR renderer API, point-in-time (default 2026-09-01), so re-runs are reproducible.
  The renderer tags each paragraph with its citation (data-title="1910.178(l)(4)(iii)"). That tag is
  written at the start of each line, so chunks carry exact paragraph-level citations.
* California: Gov. Code 12950.1 from leginfo.legislature.ca.gov (no point-in-time API).

Usage: python data/fetch_regulations.py [--date YYYY-MM-DD]
"""

import argparse
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUT = Path(__file__).parent / "regulations"
ECFR = "https://www.ecfr.gov/api/renderer/v1/content/enhanced/{date}/title-{title}?part={part}&section={section}"
CA = "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=GOV&sectionNum=12950.1"

# (title, part, section, practitioner label). Regulations rarely use everyday names: 45 CFR 164.530
# never says "HIPAA" and 1910.178 never says "forklift", so the label goes into every chunk header.
SECTIONS = [
    (29, 1910, "1910.38", "OSHA emergency action plan standard (evacuation, fire drills)"),
    (29, 1910, "1910.95", "OSHA occupational noise / hearing conservation standard"),
    (29, 1910, "1910.132", "OSHA personal protective equipment (PPE) general standard"),
    (29, 1910, "1910.134", "OSHA respiratory protection standard (respirators, fit testing, N95)"),
    (29, 1910, "1910.146", "OSHA permit-required confined spaces standard"),
    (29, 1910, "1910.147", "OSHA lockout/tagout (LOTO) standard, control of hazardous energy"),
    (29, 1910, "1910.157", "OSHA portable fire extinguishers standard"),
    (29, 1910, "1910.178", "OSHA powered industrial trucks (forklift) standard"),
    (29, 1910, "1910.1030", "OSHA bloodborne pathogens (BBP) standard"),
    (29, 1910, "1910.1200", "OSHA hazard communication (HazCom, GHS, SDS) standard"),
    (45, 164, "164.308", "HIPAA Security Rule, administrative safeguards"),
    (45, 164, "164.530", "HIPAA Privacy Rule, administrative requirements"),
]
HEADERS = {"User-Agent": "Mozilla/5.0 (learning-compliance-rag research)"}


def front_matter(**kv) -> str:
    return "---\n" + "".join(f'{k}: "{v}"\n' for k, v in kv.items()) + "---\n"


def fetch_ecfr(title, part, section, topic, date) -> str:
    url = ECFR.format(date=date, title=title, part=part, section=section)
    soup = BeautifulSoup(requests.get(url, headers=HEADERS, timeout=60).text, "html.parser")
    head = " ".join(soup.find(attrs={"data-hierarchy-metadata": True}).get_text().split())
    lines = [front_matter(doc_id=f"{title}-CFR-{section}", title=head, doc_type="regulation", status="current",
                          citation=f"{title} CFR {section}", topic=topic, source_url=url, as_of=date),
             f"# {title} CFR {head.lstrip('§ ')}\n"]
    for el in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5"]):
        text = " ".join(el.get_text().split())
        pid = el.get("data-title") or ""
        if not text or el.name.startswith("h") and (text == head or "Editorial Note" in text):
            continue
        if el.name.startswith("h"):  # appendix headings
            lines.append(f"\n## {text}\n")
        elif pid and pid.count("(") == 1:  # top-level paragraph, e.g. (l): becomes a section
            lines.append(f"\n## {title} CFR {pid}\n[{title} CFR {pid}] {text}")
        elif pid:
            lines.append(f"[{title} CFR {pid}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines) + "\n"


def fetch_ca() -> str:
    soup = BeautifulSoup(requests.get(CA, headers=HEADERS, timeout=60).text, "html.parser")
    body = soup.find(id="codeLawSectionNoHead")
    paras = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    paras = paras[next(i for i, p in enumerate(paras) if p.startswith("(a)")):]
    lines, top = [], ""
    for p in paras:  # cite as subdivision + paragraph, e.g. 12950.1(a)(1); deeper levels inherit
        marks = re.findall(r"^\s*((?:\(\w+\)\s*)+)", p)
        groups = re.findall(r"\(\w+\)", marks[0]) if marks else []
        if groups and re.fullmatch(r"\([a-z]\)", groups[0]) and groups[0] not in ("(i)", "(v)", "(x)") or \
                groups[:1] == ["(i)"] and top == "(h)":
            top, groups = groups[0], groups[1:]
        cite = top + "".join(g for g in groups if g[1].isdigit())
        lines.append(f"[Cal. Gov. Code 12950.1{cite}] {p}" if groups or cite else p)
    return (front_matter(doc_id="CA-GOV-12950.1", title="Sexual harassment prevention training", doc_type="statute",
                         status="current", citation="Cal. Gov. Code 12950.1",
                         topic="California law on sexual harassment prevention training", source_url=CA)
            + "# Cal. Gov. Code 12950.1 Sexual harassment prevention training\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-09-01")
    date = ap.parse_args().date
    OUT.mkdir(exist_ok=True)
    for t, p, s, topic in SECTIONS:
        (OUT / f"{t}-CFR-{s}.md").write_text(fetch_ecfr(t, p, s, topic, date))
        print("fetched", s)
    (OUT / "CA-GOV-12950.1.md").write_text(fetch_ca())
    print("fetched CA-GOV-12950.1")
