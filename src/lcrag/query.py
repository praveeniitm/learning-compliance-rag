"""Query understanding: domain vocabulary expansion.

Practitioners and regulations use different words for the same thing. Employees say
"forklift", while 29 CFR 1910.178 says "powered industrial truck" and never uses the word
"forklift". A bi-encoder only partly bridges that gap, and BM25 cannot bridge it at all. A small
curated glossary, which a compliance team would maintain, appends the regulatory term to the
query. Expansion is additive (the original words are kept), so it does not hurt queries that
already use the regulatory term.
"""

from __future__ import annotations

import re

GLOSSARY: dict[str, str] = {
    r"\bforklifts?\b|\bfork lifts?\b|\breach trucks?\b|\bpallet jacks?\b": "powered industrial truck operator",
    r"\bloto\b|\block ?out\b|\btag ?out\b": "lockout tagout control of hazardous energy",
    r"\bbbp\b|\bneedlestick\b|\bsharps\b": "bloodborne pathogens occupational exposure",
    r"\bphi\b|\bpatient (data|records|information)\b": "protected health information HIPAA",
    r"\bfit[- ]?test(s|ing)?\b|\bn95\b|\bmasks?\b": "respirator fit testing respiratory protection",
    r"\bhazcom\b|\bsds\b|\bsafety data sheets?\b|\bghs\b": "hazard communication",
    r"\bhearing\b|\bnoise\b|\bearplugs?\b": "hearing conservation occupational noise",
    r"\bconfined spaces?\b|\btanks?\b|\bvaults?\b": "permit-required confined space",
    r"\bextinguishers?\b": "portable fire extinguisher",
    r"\bevacuation\b|\bfire drill\b|\bemergency plan\b": "emergency action plan",
    r"\bsexual harassment\b|\bharassment\b": "harassment prevention training abusive conduct",
    r"\brecert(ification)?\b|\brefresher\b|\bretrain(ing)?\b": "recertification interval refresher training",
    r"\blate\b|\bmissed\b|\bpast due\b|\boverdue\b": "overdue escalation grace period",
    r"\bnew hires?\b|\bonboarding\b": "new hire assignment window",
}
_COMPILED = [(re.compile(p, re.I), exp) for p, exp in GLOSSARY.items()]


TOPIC_PATTERNS: dict[str, str] = {
    "bloodborne pathogens": r"bloodborne|\bbbp\b|needlestick|sharps",
    "HIPAA": r"hipaa|privacy|\bphi\b|security awareness|protected health",
    "harassment prevention": r"harass|abusive conduct",
    "forklift": r"forklift|fork lift|powered industrial|pallet jack|reach truck|\bpit\b",
    "respiratory protection": r"respirat|fit[- ]?test|\bn95\b",
    "lockout/tagout": r"lockout|tagout|\bloto\b|hazardous energy",
    "hazard communication": r"hazcom|hazard communication|\bsds\b|\bghs\b|chemical",
    "fire extinguishers": r"extinguisher",
    "hearing conservation": r"hearing|noise",
    "confined spaces": r"confined space",
    "emergency action plan": r"emergency action|evacuation|\beap\b",
    "code of conduct": r"code of conduct|ethics",
}
_TRAINING_Q = re.compile(r"\b(training|refresher|recert\w*|retrain\w*|course|how often|interval)\b", re.I)
_SPECIFIC = re.compile(r"\b[A-Z]{2,4}(-[A-Z0-9]+)+\b|\d{2,5}\.\d+|\bpolicy\b|\bgrace\b|\bescalat|\boverdue\b|"
                       r"\bnew hire|\bsupervisor|\baudit|\bequivalen|\bexempt|\brecord", re.I)


def detect_topics(query: str) -> list[str]:
    return [t for t, p in TOPIC_PATTERNS.items() if re.search(p, query, re.I)]


def is_underspecified(query: str) -> bool:
    """Rule-based check: asks about a training requirement but names no topic, course or rule."""
    return bool(_TRAINING_Q.search(query)) and not detect_topics(query) and not _SPECIFIC.search(query)


def expand(query: str) -> str:
    extra = [exp for rx, exp in _COMPILED if rx.search(query) and exp.lower() not in query.lower()]
    return query if not extra else f"{query} ({'; '.join(dict.fromkeys(extra))})"
