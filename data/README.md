# Corpus

The corpus combines **real public regulations** (the legal source of truth) with **synthetic internal documents** for a fictional organization. That mirrors how compliance-training questions are answered in practice: the answer usually lives in the gap between what the law requires and what the organization has decided to enforce.

## 1. Public regulations (`fetch_regulations.py` → `data/regulations/*.md`, committed)

| Doc ID | Citation | Why it is included |
|---|---|---|
| 29-CFR-1910.38 | Emergency action plans | Plan review/training triggers |
| 29-CFR-1910.95 | Occupational noise exposure | Annual hearing-conservation training |
| 29-CFR-1910.132 | PPE, general requirements | Training and retraining triggers |
| 29-CFR-1910.134 | Respiratory protection | Annual training + fit testing |
| 29-CFR-1910.146 | Permit-required confined spaces | Role-based training |
| 29-CFR-1910.147 | Lockout/tagout | Authorized vs affected training, retraining |
| 29-CFR-1910.157 | Portable fire extinguishers | Annual education |
| 29-CFR-1910.178 | Powered industrial trucks | Operator training, 3-year evaluation |
| 29-CFR-1910.1030 | Bloodborne pathogens | Annual training, record retention |
| 29-CFR-1910.1200 | Hazard communication | Training on new hazards; large appendices act as realistic distractors |
| 45-CFR-164.308 | HIPAA Security Rule | Security awareness program |
| 45-CFR-164.530 | HIPAA Privacy Rule | Workforce training, documentation retention |
| CA-GOV-12950.1 | Cal. Gov. Code § 12950.1 | Harassment prevention hours, frequency, deadlines |

- **Source:** the [eCFR renderer API](https://www.ecfr.gov/developers/documentation/api/v1), queried **point-in-time** (default `2026-09-01`), so a re-run returns the same text even after amendments. eCFR tags every paragraph with its citation, and the fetcher writes that tag at the start of each line (`[29 CFR 1910.178(l)(4)(iii)] ...`), so chunks carry exact paragraph-level citations. The California statute comes from [leginfo.legislature.ca.gov](https://leginfo.legislature.ca.gov), which has no point-in-time API.
- **License:** U.S. federal regulations are public domain. California statutes are public law.
- **Selection rule:** only sections containing explicit *training* obligations (who, what, how often). That bounds the corpus to the question space and keeps the non-training text as realistic hard negatives, for example 1910.1200's roughly 300k characters of classification appendices.

## 2. Synthetic organization (`generate_synthetic.py` → `data/synthetic/docs/`, committed)

"Northwind Manufacturing & Health" is a fictional company with 4,800 employees across three plants or distribution centers, two occupational-health clinics and a headquarters. That mix puts OSHA general-industry rules, HIPAA, and a California statute in scope at once, which is common in real organizations and forces cross-source reasoning.

| Doc ID | Type | What it contributes |
|---|---|---|
| POL-CT-001-v3.2 | Policy (current) | Precedence rules, windows, intervals, grace periods, records |
| POL-CT-001-v3.1 | Policy (**superseded**) | Older, conflicting values; tests version awareness |
| CAT-2026 | Course catalog | Course ↔ requirement mapping, durations, recertification, retired courses |
| MAT-ROLE-2026 | Role matrix | Job code → curriculum |
| SOP-LMS-014 / -021 / -030 | LMS SOPs | Assignment rules, escalation, equivalency |
| SOP-EHS-007 / -012 | EHS SOPs | Forklift certification, respirator fit testing |
| AUD-2026-Q2 | Audit memo | Findings with numbers, owners, root causes |
| FAQ-HR-HAR | FAQ | Manager-facing Q&A |
| EML-* | E-mails | Informal communication, including one **incorrect** claim and one with a **hidden prompt injection** (EML-2026-05-lms-tip) |
| RN-LMS-2026-08 | Release notes | Most recent change; used to demo index freshness |

### Access levels (front matter `access`)

Regulations and most internal documents are `all`. Restricted: AUD-2026-Q2 (compliance, ehs, lms_admin), EML-2026-03-FRE-forklift (compliance, ehs), RN-LMS-2026-08 (compliance, ehs, lms_admin), SOP-LMS-014 (compliance, lms_admin) and the superseded POL-CT-001-v3.1 (compliance only). Retrieval enforces these per role.

### Generation method and assumptions

- **Single source of truth.** Every number (intervals, windows, headcounts, audit figures) lives in `synthetic/facts.yaml`. Templates in `synthetic/templates/` render them into prose and tables, and the evaluation set is written against the same facts.
- **Deterministic.** Rendering uses no randomness and no LLM, so re-running produces identical files.
- **Legally consistent by construction.** Each internal rule was checked against the fetched regulation text. The organization's rules are equal to or stricter than the law, never looser, as a real compliance office would require.
- **Injected imperfections** (listed in `facts.yaml` → `injected_conflicts`):
  - A superseded policy version still sits in the corpus.
  - A supervisor's e-mail wrongly claims forklift re-evaluation is every 24 months, and that a "safety video" suffices after a rack strike.
  - A known configuration gap: a job code missing its HIPAA courses.

### Gap to real-world data

Real enterprise corpora are messier:
- scanned PDFs and OCR errors
- inconsistent headings
- duplicated and near-duplicated SOPs across sites
- far more e-mail volume
- policy text that is vaguer than these templates

This corpus is cleanly structured, which flatters structure-aware chunking. Results should be read as an upper bound for that component. The regulations are the real, unmodified text.
