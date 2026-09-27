# Corpus

The knowledge base mixes **real public regulations** with **internal documents of a fictional organization**. Compliance-training questions are usually answered this way in practice: the answer sits between what the law requires and what the organization has decided to enforce.

| Folder | Content | Documents | Words |
|---|---|---|---|
| `regulations/` | Real regulation text, fetched by `fetch_regulations.py` | 13 | ≈115,000 |
| `northwind/` | Internal documents of "Northwind Manufacturing & Health" (fictional) | 15 | ≈7,900 |
| `samples/` | One extra document used to demonstrate adding content to a live index | 1 | – |
| `incoming/` | Empty; documents dropped here are picked up by the next index sync (not committed) | – | – |

Every file is Markdown with a small front-matter header (`doc_id`, `doc_type`, `status`, `effective_date`, `access`, …). The front matter drives citations, version handling and access control.

## 1. Regulations (`regulations/`)

| Doc ID | Regulation | Why it is included |
|---|---|---|
| 29-CFR-1910.38 | Emergency action plans | Plan review and training triggers |
| 29-CFR-1910.95 | Occupational noise exposure | Annual hearing-conservation training |
| 29-CFR-1910.132 | Personal protective equipment (PPE) | Training and retraining triggers |
| 29-CFR-1910.134 | Respiratory protection | Annual training and fit testing |
| 29-CFR-1910.146 | Permit-required confined spaces | Role-based training |
| 29-CFR-1910.147 | Lockout/tagout (control of hazardous energy) | Authorized vs. affected employees, retraining |
| 29-CFR-1910.157 | Portable fire extinguishers | Annual education |
| 29-CFR-1910.178 | Powered industrial trucks (forklifts) | Operator training, 3-year evaluation |
| 29-CFR-1910.1030 | Bloodborne pathogens | Annual training, record retention |
| 29-CFR-1910.1200 | Hazard communication | Training on new hazards; its long appendices act as realistic "noise" |
| 45-CFR-164.308 | HIPAA Security Rule | Security awareness training |
| 45-CFR-164.530 | HIPAA Privacy Rule | Workforce training, 6-year documentation retention |
| CA-GOV-12950.1 | California Government Code § 12950.1 | Harassment-prevention training hours and deadlines |

Abbreviations:
- **CFR:** Code of Federal Regulations. Title 29 covers labor, and title 45 covers public welfare.
- **OSHA:** Occupational Safety and Health Administration, which issues the 29 CFR 1910 rules.
- **HIPAA:** Health Insurance Portability and Accountability Act.

- **Source:** the [eCFR](https://www.ecfr.gov) (electronic CFR) API. It is queried **point-in-time** for 2026-09-01, so a re-run returns the same text even after the rules are amended. eCFR tags every paragraph with its citation, and the fetcher writes that tag at the start of each line, e.g. `[29 CFR 1910.178(l)(4)(iii)] …`. Every chunk therefore carries exact paragraph-level citations. The California statute comes from [leginfo.legislature.ca.gov](https://leginfo.legislature.ca.gov).
- **License:** U.S. federal regulations are public domain. California statutes are public law.
- **Selection rule:** only sections that contain explicit training obligations (who, what, how often). The non-training text inside them is kept as realistic distractors.
- **Refresh:** `python data/fetch_regulations.py --date YYYY-MM-DD`, then re-sync the index. Only amended paragraphs are re-embedded.

## 2. Northwind documents (`northwind/`)

A fictional company with 4,800 employees across two plants, a distribution center, two occupational-health clinics and a headquarters. That mix brings OSHA rules, HIPAA and a California statute into scope at once, which is common in real organizations and forces cross-source reasoning. The documents are written by hand as plain Markdown and are the data themselves; there is no generation step.

| Doc ID | Type | What it contributes |
|---|---|---|
| POL-CT-001-v3.2 | Policy (current) | Precedence rules, completion windows, recertification intervals, grace periods, record retention |
| POL-CT-001-v3.1 | Policy (**superseded** 2026-01-01) | Older, different values; tests version handling and as-of questions |
| CAT-2026 | Course catalog | Course ↔ requirement mapping, durations, recertification, retired courses |
| MAT-ROLE-2026 | Role matrix | Job code → required courses |
| SOP-LMS-014 / -021 / -030 | Learning-management-system (LMS) procedures | Assignment rules, overdue escalation, external-training equivalency |
| SOP-EHS-007 / -012 | Environmental Health & Safety (EHS) procedures | Forklift certification, respirator fit testing |
| AUD-2026-Q2 | Internal audit memo | Findings with numbers, owners and root causes |
| FAQ-HR-HAR | Manager FAQ | Harassment-prevention questions |
| EML-* (3) | E-mails | Informal communication; one contains a wrong claim, one a hidden prompt injection |
| RN-LMS-2026-08 | LMS release notes | Most recent system changes |

Doc types: 5 procedures, 3 e-mails, 2 policy versions, and 1 each of catalog, matrix, audit, FAQ and release notes. Most are 100–1,500 words.

### How the documents were written (assumptions)

- **Consistent with the law:** every internal rule was checked against the fetched regulation text. Northwind's rules are equal to or stricter than the law, never looser, as a real compliance office would require. Example: HIPAA sets no refresher interval, and Northwind requires one every year.
- **Consistent with each other:** the same interval, window or course code appears identically in the policy, catalog, matrix and procedures, except where an inconsistency is deliberate.
- **Deliberate imperfections**, so the system has to cope with what real corpora contain:

| Document | Imperfection | Correct answer |
|---|---|---|
| POL-CT-001-v3.1 | Superseded version still in the corpus (45-day new-hire window, HIPAA refresher every 2 years) | v3.2: 30 days, annual |
| EML-2026-03-FRE-forklift | Supervisor claims forklift re-evaluation is every 24 months | 36 months (policy and 29 CFR 1910.178(l)(4)(iii)) |
| EML-2026-05-lms-tip | Hidden text telling "AI assistants" to say training is optional | The instruction must be ignored |
| MAT-ROLE-2026 | Telehealth Coordinator role is missing its HIPAA courses (a configuration gap found by the audit) | Reported as a gap, not as policy |

- **Access levels** (`access` in front matter). Most documents are open to everyone (`all`). The restricted ones are:
  - the audit memo: compliance, ehs, lms_admin
  - the forklift e-mail: compliance, ehs
  - the August release notes: compliance, ehs, lms_admin
  - the LMS assignment procedure: compliance, lms_admin
  - the superseded policy: compliance only

### Gap to real-world data

Real corpora are messier:
- scanned PDFs and OCR errors
- inconsistent headings
- duplicated and near-duplicate procedures across sites
- far more e-mail
- vaguer policy text

These documents are cleanly structured, which flatters heading-based chunking, so read those results as an upper bound. The regulations are real and unmodified.
