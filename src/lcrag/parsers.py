"""Parse raw sources into structured Documents made of citable Blocks.

Three formats:
  * eCFR XML        -> paragraphs addressed as (a)(1)(i)(A)(1)(i), plus appendix headings/tables
  * CA leginfo HTML -> subdivisions addressed as (a)(1)(A)
  * Markdown with YAML front matter (synthetic internal documents) -> headings and numbered clauses

Keeping the paragraph address is the core design decision of the ingestion layer: answers in this
domain are only useful if they cite *where* a requirement lives (e.g. 29 CFR 1910.178(l)(4)(iii)),
and evaluation relevance is defined on these addresses rather than on chunk ids, so different
chunking strategies can be compared on the same ground truth.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import yaml
from lxml import etree
from lxml import html as lxml_html

from .schema import Block, Document

# --------------------------------------------------------------------------------------------
# CFR-style paragraph addressing
# --------------------------------------------------------------------------------------------
# Levels used by the Federal Register / eCFR: (a) -> (1) -> (i) -> (A) -> (1 italic) -> (i italic)
_ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv",
          "xvi", "xvii", "xviii", "xix", "xx", "xxi", "xxii", "xxiii", "xxiv", "xxv"]
_ITAL_OPEN, _ITAL_CLOSE = "\x01", "\x02"
_MARKER_RE = re.compile(r"\s*[—–-]?\s*(\x01)?\((\w{1,5})\)(\x02)?")
_ITAL_TITLE_RE = re.compile(r"\s*\x01([^\x02]{1,160})\x02")
_PLAIN_TITLE_RE = re.compile(r"\s*([A-Z][^.()\x01\x02]{0,80}\.)(?=\s*[—–-]?\s*\x01?\(\w{1,5}\))")


def _next_letter(cur: str | None) -> str:
    if not cur:
        return "a"
    if len(cur) == 1:
        return "aa" if cur == "z" else chr(ord(cur) + 1)
    return chr(ord(cur[0]) + 1) * 2  # (aa) -> (bb)


def _next_roman(cur: str | None) -> str:
    if not cur:
        return "i"
    try:
        return _ROMAN[_ROMAN.index(cur) + 1]
    except (ValueError, IndexError):
        return ""


# Paragraph level order differs by jurisdiction:
#   Federal Register / eCFR:  (a) (1) (i) (A) (1 italic) (i italic)
#   California codes:         (a) (1) (A) (i)
CFR_SCHEME = ("lower", "digit", "roman", "upper", "idigit", "iroman")
CA_SCHEME = ("lower", "digit", "upper", "roman")


class ParagraphTracker:
    """Tracks the current paragraph address while walking a regulation in document order."""

    def __init__(self, scheme: tuple[str, ...] = CFR_SCHEME) -> None:
        self.scheme = scheme
        self.depth = len(scheme)
        self.path: list[str | None] = [None] * self.depth
        self.titles: list[str | None] = [None] * self.depth

    def reset(self) -> None:
        self.__init__(self.scheme)

    def _lvl(self, kind: str) -> int:
        return self.scheme.index(kind) if kind in self.scheme else self.depth - 1

    def _classify(self, tok: str, italic: bool, has_title: bool) -> int:
        if italic:
            return self._lvl("idigit" if tok.isdigit() else "iroman")
        if tok.isdigit():
            return self._lvl("digit")
        if tok.isupper():
            return self._lvl("upper")
        lower, roman = self._lvl("lower"), self._lvl("roman")
        is_roman = tok in _ROMAN
        is_letter = len(tok) == 1 or (len(tok) == 2 and tok[0] == tok[1])
        if is_roman and not is_letter:
            return roman
        if is_letter and not is_roman:
            return lower
        # Ambiguous: (i), (v), (x) can be a letter after (h)/(u)/(w) or a roman numeral.
        # A roman numeral is only plausible if its parent level is currently open.
        exp_letter = _next_letter(self.path[lower])
        parent_open = self.path[roman - 1] is not None
        exp_roman = _next_roman(self.path[roman]) if parent_open else None
        if tok == exp_roman and tok != exp_letter:
            return roman
        if tok == exp_letter and tok != exp_roman:
            return lower
        if tok == exp_letter == exp_roman:
            return lower if has_title else roman  # top-level paragraphs usually carry a title
        return roman if parent_open else lower

    def _set(self, level: int, tok: str, title: str | None) -> None:
        self.path[level] = tok
        self.titles[level] = title
        for deeper in range(level + 1, self.depth):
            self.path[deeper] = None
            self.titles[deeper] = None

    def consume(self, marked: str) -> bool:
        """Consume leading paragraph markers of `marked` (italics as \\x01..\\x02).

        Returns True if at least one marker was found (i.e. a new paragraph starts).
        Handles chained markers such as "(l) Operator training. (1) <I>Safe operation.</I> (i) ...".
        """
        pos, found = 0, False
        while True:
            m = _MARKER_RE.match(marked, pos)
            if not m:
                break
            tok, italic = m.group(2), bool(m.group(1))
            if not (tok.isdigit() or tok.isalpha()) or len(tok) > 5:
                break
            after = m.end()
            t = _ITAL_TITLE_RE.match(marked, after) or _PLAIN_TITLE_RE.match(marked, after)
            title = t.group(1).strip().rstrip(".").strip() if t else None
            self._set(self._classify(tok, italic, title is not None), tok, title)
            found = True
            pos = t.end() if t else after
        return found

    @property
    def anchor(self) -> str:
        return "".join(f"({p})" for p in self.path if p)

    @property
    def top(self) -> str:
        return f"({self.path[0]})" if self.path[0] else ""

    def trail(self) -> list[str]:
        out, prefix = [], ""
        for p, t in zip(self.path, self.titles):
            if not p:
                continue
            prefix += f"({p})"
            if t:
                out.append(f"{prefix} {t}")
        return out


def _clean(text: str) -> str:
    text = text.replace(_ITAL_OPEN, "").replace(_ITAL_CLOSE, "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _inner_marked(el: etree._Element) -> str:
    """Inner text of an element with <I> spans marked, other tags stripped."""
    raw = etree.tostring(el, encoding="unicode", with_tail=False)
    raw = re.sub(r"^<[^>]+>|</[^>]+>$", "", raw.strip())
    raw = re.sub(r"<I>", _ITAL_OPEN, raw)
    raw = re.sub(r"</I>", _ITAL_CLOSE, raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return html.unescape(raw)


def _table_text(table: etree._Element) -> str:
    rows = []
    for tr in table.iter("TR", "tr"):
        cells = [_clean(" ".join(c.itertext())) for c in tr if c.tag in ("TD", "TH", "td", "th")]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


# --------------------------------------------------------------------------------------------
# eCFR XML
# --------------------------------------------------------------------------------------------
def parse_ecfr(path: Path, entry: dict) -> Document:
    root = etree.fromstring(path.read_bytes())
    head = _clean(root.findtext("HEAD") or entry["citation"])
    title = re.sub(r"^§\s*[\d.]+\s*", "", head).rstrip(".")
    tracker = ParagraphTracker()
    blocks: list[Block] = []
    appendix: str | None = None
    sub_heading: str | None = None

    def base_trail() -> list[str]:
        t = [head.rstrip(".")]
        if appendix:
            t.append(appendix)
            if sub_heading:
                t.append(sub_heading)
            return t
        return t + tracker.trail()

    for el in root.iter():
        tag = el.tag if isinstance(el.tag, str) else ""
        if tag in ("HD1", "HD2", "HD3"):
            text = _clean("".join(el.itertext()))
            if text.lower().startswith("appendix"):
                appendix, sub_heading = text, None
                tracker.reset()
            elif appendix:
                sub_heading = text
            blocks.append(Block("heading", text, anchor=_appendix_anchor(appendix) or tracker.anchor,
                                trail=base_trail(), top=_appendix_anchor(appendix) or tracker.top,
                                appendix=bool(appendix)))
        elif tag in ("P", "FP"):
            marked = _inner_marked(el)
            if not appendix:
                tracker.consume(marked)
            text = _clean(marked)
            if not text:
                continue
            anchor = _appendix_anchor(appendix) if appendix else tracker.anchor
            top = _appendix_anchor(appendix) if appendix else tracker.top
            blocks.append(Block("para", text, anchor=anchor, trail=base_trail(), top=top, appendix=bool(appendix)))
        elif tag in ("TABLE",):
            text = _table_text(el)
            if text:
                anchor = _appendix_anchor(appendix) if appendix else tracker.anchor
                top = _appendix_anchor(appendix) if appendix else tracker.top
                blocks.append(Block("table", text, anchor=anchor, trail=base_trail(), top=top, appendix=bool(appendix)))

    return Document(
        doc_id=entry["doc_id"],
        title=title,
        doc_type="regulation",
        citation=entry["citation"],
        blocks=blocks,
        meta={k: entry[k] for k in ("source_url", "as_of", "sha256") if k in entry} | {"status": "current",
                                                                                       "authority": 1},
    )


def _appendix_anchor(appendix: str | None) -> str:
    if not appendix:
        return ""
    m = re.match(r"(Appendix\s+[A-Z0-9]+)", appendix)
    return m.group(1) if m else appendix[:40]


# --------------------------------------------------------------------------------------------
# California statute HTML
# --------------------------------------------------------------------------------------------
def parse_ca_statute(path: Path, entry: dict) -> Document:
    tree = lxml_html.fromstring(path.read_text(encoding="utf-8"))
    paras = [_clean(p.text_content()) for p in tree.iter("p")]
    paras = [p for p in paras if p]
    # The page repeats the hierarchy (title/division/chapter) before the section body.
    start = next((i for i, p in enumerate(paras) if re.match(r"^\(a\)", p)), 0)
    tracker = ParagraphTracker(CA_SCHEME)
    blocks: list[Block] = []
    head = entry["citation"]
    for p in paras[start:]:
        if p.startswith("(Amended by") or p.startswith("(Added by"):
            blocks.append(Block("para", p, anchor="history", trail=[head], top="history"))
            continue
        tracker.consume(p)
        blocks.append(Block("para", p, anchor=tracker.anchor, trail=[head] + tracker.trail(), top=tracker.top))
    return Document(
        doc_id=entry["doc_id"],
        title="Sexual harassment prevention training (California)",
        doc_type="statute",
        citation=entry["citation"],
        blocks=blocks,
        meta={k: entry[k] for k in ("source_url", "as_of", "sha256") if k in entry} | {"status": "current",
                                                                                       "authority": 1},
    )


# --------------------------------------------------------------------------------------------
# Markdown with front matter (internal documents)
# --------------------------------------------------------------------------------------------
_AUTHORITY = {"policy": 2, "sop": 3, "catalog": 3, "matrix": 3, "audit": 4, "faq": 4, "release_notes": 4,
              "email": 5}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_CLAUSE_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\s")
_FAQ_RE = re.compile(r"^\*\*(Q\d+)\.")
_ID_IN_HEADING_RE = re.compile(r"^(\d+)\.\s|^([A-Z]{1,4}(?:-[A-Z0-9]+)+|F-\d+)\b")


def _heading_anchor(text: str) -> str:
    m = _ID_IN_HEADING_RE.match(text)
    if m:
        return m.group(1) or m.group(2)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def parse_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    fm_match = re.match(r"^---\n(.*?)\n---\n", raw, flags=re.S)
    meta = yaml.safe_load(fm_match.group(1)) if fm_match else {}
    body = raw[fm_match.end():] if fm_match else raw
    meta = {k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in meta.items()}

    doc_title = meta.get("title", path.stem)
    headings: dict[int, tuple[str, str]] = {}  # level -> (text, anchor)
    blocks: list[Block] = []

    def trail() -> list[str]:
        return [f"{doc_title}"] + [headings[lvl][0] for lvl in sorted(headings) if lvl >= 2]

    def section_anchor() -> str:
        levels = [lvl for lvl in sorted(headings) if lvl >= 2]
        return headings[levels[-1]][1] if levels else ""

    def top() -> str:
        levels = [lvl for lvl in sorted(headings) if 2 <= lvl <= 3]
        return headings[levels[-1]][1] if levels else ""

    for para in re.split(r"\n\s*\n", body):
        para = para.strip("\n")
        if not para.strip():
            continue
        m = _HEADING_RE.match(para.strip())
        if m and "\n" not in para.strip():
            level, text = len(m.group(1)), m.group(2).strip()
            for deeper in [lvl for lvl in headings if lvl >= level]:
                del headings[deeper]
            headings[level] = (text, _heading_anchor(text))
            if level >= 2:
                blocks.append(Block("heading", text, anchor=section_anchor(), trail=trail(), top=top()))
            continue
        lines = [ln for ln in para.splitlines() if ln.strip()]
        is_table = all(ln.lstrip().startswith("|") for ln in lines)
        if is_table:
            rows = [ln for ln in lines if not re.match(r"^\s*\|[\s|:-]+\|\s*$", ln)]
            text = "\n".join(" | ".join(c.strip() for c in r.strip().strip("|").split("|")) for r in rows)
            blocks.append(Block("table", text, anchor=section_anchor(), trail=trail(), top=top()))
            continue
        text = re.sub(r"[ \t]+", " ", para.strip())
        cm = _CLAUSE_RE.match(text) or _FAQ_RE.match(text)
        anchor = cm.group(1) if cm else section_anchor()
        blocks.append(Block("para", text, anchor=anchor, trail=trail(), top=top()))

    doc_type = meta.get("doc_type", "document")
    meta["authority"] = _AUTHORITY.get(doc_type, 5)
    doc_id = meta.pop("doc_id", path.stem)
    return Document(doc_id=doc_id, title=doc_title, doc_type=doc_type, citation=doc_id, blocks=blocks, meta=meta)


# --------------------------------------------------------------------------------------------
def load_corpus(raw_dir: Path, synth_dir: Path) -> list[Document]:
    docs: list[Document] = []
    manifest_path = raw_dir / "manifest.json"
    if manifest_path.exists():
        for entry in json.loads(manifest_path.read_text()):
            p = raw_dir / entry["path"]
            if entry["format"] == "ecfr_xml":
                docs.append(parse_ecfr(p, entry))
            elif entry["format"] == "ca_leginfo_html":
                docs.append(parse_ca_statute(p, entry))
    for p in sorted(synth_dir.glob("*.md")):
        docs.append(parse_markdown(p))
    return docs
