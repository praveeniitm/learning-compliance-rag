"""Render the synthetic organization's documents from facts.yaml + Jinja templates.

Design:
  * facts.yaml is the single source of truth (intervals, courses, roles, audit findings).
    Tables and prose are rendered from it, so documents agree with each other, and the
    evaluation set can be written against the same facts.
  * Realistic imperfections are injected on purpose and listed in facts.yaml under
    `injected_conflicts`: a superseded policy version that is still in the corpus, and an
    informal e-mail that contradicts current policy. A good system must prefer current,
    authoritative sources over those.
  * Output is deterministic (no randomness, no LLM), so re-running produces identical files.

Usage:
    python data/generate_synthetic.py
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

HERE = Path(__file__).resolve().parent
SYNTH = HERE / "synthetic"
OUT = SYNTH / "docs"


def build_context(facts: dict) -> dict:
    course_roles: dict[str, list[str]] = defaultdict(list)
    for role in facts["roles"]:
        for code in role["curriculum"]:
            course_roles[code].append(role["code"])
    return {
        **facts,
        "req_by_key": {r["key"]: r for r in facts["requirements"]},
        "course_by_code": {c["code"]: c for c in facts["courses"]},
        "course_roles": course_roles,
    }


def main() -> None:
    facts = yaml.safe_load((SYNTH / "facts.yaml").read_text(encoding="utf-8"))
    ctx = build_context(facts)
    env = Environment(
        loader=FileSystemLoader(SYNTH / "templates"),
        undefined=StrictUndefined,  # fail loudly on a typo instead of rendering blanks
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.md"):
        old.unlink()

    rendered: list[tuple[str, str]] = []
    policy_tpl = env.get_template("POL-CT-001.md.j2")
    rendered.append(("POL-CT-001-v3.2.md", policy_tpl.render(**ctx, p=facts["policy"]["current"], status="current")))
    rendered.append(("POL-CT-001-v3.1.md", policy_tpl.render(**ctx, p=facts["policy"]["superseded"], status="superseded")))

    for tpl_name in sorted(env.list_templates()):
        if tpl_name.startswith("POL-CT-001"):
            continue
        rendered.append((tpl_name.removesuffix(".j2"), env.get_template(tpl_name).render(**ctx)))

    for name, text in rendered:
        # Jinja's trim_blocks can swallow the blank line before a heading; restore it so the
        # Markdown stays valid and the structure-aware chunker sees clean section breaks.
        text = re.sub(r"(?<=[^\n])\n(#{1,6} )", r"\n\n\1", text)
        (OUT / name).write_text(text, encoding="utf-8")
        print(f"rendered {name:40s} {len(text):>7,d} chars")
    print(f"{len(rendered)} documents -> {OUT}")


if __name__ == "__main__":
    main()
