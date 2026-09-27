# Example questions end to end

Each example shows the agent's searches, the sources it used, the answer, and the faithfulness annotation from the judge model.

## q03 (law_vs_policy): Is the annual HIPAA refresher a legal requirement or a Northwind rule?

**Status:** answered

**Searches:** `regulation`: annual HIPAA refresher training requirement; `internal`: annual HIPAA refresher training requirement

**Retrieved context (top 3 of 10):**

> [S1] [45 CFR 164.530] § 164.530 Administrative requirements. (HIPAA Privacy Rule, administrative requirements) 45 CFR 164.530(b) [45 CFR 164.530(b)] (b) [45 CFR 164.530(b)(1)] (1) Standard: Training. A covered entity must train all members of its workforce on the policies and procedures with respect to p…
> [S2] [45 CFR 164.530] § 164.530 Administrative requirements. (HIPAA Privacy Rule, administrative requirements) 45 CFR 164.530(b) [45 CFR 164.530(b)(2)(ii)] (ii) A covered entity must document that the training as described in paragraph (b)(2)(i) of this section has been provided, as required by paragraph…
> [S3] [45 CFR 164.530] § 164.530 Administrative requirements. (HIPAA Privacy Rule, administrative requirements) 45 CFR 164.530(j) [45 CFR 164.530(j)] (j) [45 CFR 164.530(j)(1)] (1) Standard: Documentation. A covered entity must: [45 CFR 164.530(j)(1)(i)] (i) Maintain the policies and procedures provided f…

**Answer:** The HIPAA refresher training interval of every 12 months is a Northwind policy requirement, not a legal requirement, as adopted after the 2025 risk assessment [S6][S7][S8][S9].

**Faithfulness:** 1.00 of statements supported

## q27 (multi_doc): Which courses must a Clinic Nurse complete, and which of them recur every 12 months?

**Status:** unverified

**Searches:** `internal`: Clinic Nurse required courses; `internal`: Clinic Nurse course recertification interval

**Retrieved context (top 3 of 8):**

> [S1] [MAT-ROLE-2026] Role-to-Curriculum Matrix 2026 | matrix v2026.3, effective 2026-03-15 Matrix | CLN-RN | Clinic Nurse | FCL, CLB | no | EHS-BBP-100, PRV-HIP-110, PRV-HIP-120, EHS-RSP-110, EHS-HC-100, EHS-EAP-100, HR-HAR-100, ETH-COC-100 | | CLN-MA | Medical Assistant | FCL, CLB | no | EHS-BBP-100, PR…
> [S2] [MAT-ROLE-2026] Role-to-Curriculum Matrix 2026 | matrix v2026.3, effective 2026-03-15 Role Notes > CLN-RN Clinic Nurse Employees in job code CLN-RN (Clinic Nurse) at FCL, CLB are assigned: EHS-BBP-100 Bloodborne Pathogens Annual; PRV-HIP-110 HIPAA Privacy Essentials; PRV-HIP-120 HIPAA Security Aware…
> [S3] [MAT-ROLE-2026] Role-to-Curriculum Matrix 2026 | matrix v2026.3, effective 2026-03-15 Role Notes > CLN-MA Medical Assistant Employees in job code CLN-MA (Medical Assistant) at FCL, CLB are assigned: EHS-BBP-100 Bloodborne Pathogens Annual; PRV-HIP-110 HIPAA Privacy Essentials; PRV-HIP-120 HIPAA Secu…

**Answer:** A Clinic Nurse (CLN-RN) must complete the following courses: EHS-BBP-100 Bloodborne Pathogens Annual, PRV-HIP-110 HIPAA Privacy Essentials, PRV-HIP-120 HIPAA Security Awareness, EHS-RSP-110 Respiratory Protection and Fit Test, EHS-HC-100 Hazard Communication and GHS Labels, EHS-EAP-100 Site Emergency Action Plan, HR-HAR-100 Harassment Prevention for Employees, and ETH-COC-100 Code of Conduct [S2][S3][S4][S5]. The courses that recur every 12 months are: EHS-BBP-100 Bloodborne Pathogens, PRV-HIP-110 HIPAA Privacy Essentials, PRV-HIP-120 HIPAA Security Awareness, and EHS-EAP-100 Site Emergency Action Plan [S4][S5][S6][S7].

**Faithfulness:** 0.50 of statements supported; unsupported: ['The courses that recur every 12 months are: EHS-BBP-100 Bloodborne Pathogens, PRV-HIP-110 HIPAA Privacy Essentials, PRV-HIP-120 HIPAA Security Awareness, and EHS-EAP-100 Site Emergency Action Plan.']

## q54 (conflict): A Fresno supervisor says forklift re-evaluations are every 24 months now. Is that right?

**Status:** out_of_scope

**Searches:** `regulation`: forklift re-evaluation interval; `internal`: forklift re-evaluation interval Fresno supervisor

**Retrieved context (top 3 of 10):**

> [S1] [29 CFR 1910.178] § 1910.178 Powered industrial trucks. (OSHA powered industrial trucks (forklift) standard) 29 CFR 1910.178(l) [29 CFR 1910.178(l)(4)(ii)(E)] (E) A condition in the workplace changes in a manner that could affect safe operation of the truck. [29 CFR 1910.178(l)(4)(iii)] (iii) An eva…
> [S2] [29 CFR 1910.178] § 1910.178 Powered industrial trucks. (OSHA powered industrial trucks (forklift) standard) 29 CFR 1910.178(l) [29 CFR 1910.178(l)(3)(ii)(I)] (I) Other unique or potentially hazardous environmental conditions in the workplace that could affect safe operation. [29 CFR 1910.178(l)(3)(…
> [S3] [29 CFR 1910.178] § 1910.178 Powered industrial trucks. (OSHA powered industrial trucks (forklift) standard) 29 CFR 1910.178(l) [29 CFR 1910.178(l)(8)] (8) Appendix A to this section provides non-mandatory guidance to assist employers in implementing this paragraph (l). This appendix does not add to…

**Answer:** No, that is not correct. According to the current SOP (SOP-EHS-007), the practical re-evaluation for forklift operators is required every 36 months, not 24 months [S7][S9]. The email from Fresno indicates a 24-month re-evaluation cycle, but this is not aligned with the current policy or SOP, which specify a 36-month interval.

**Faithfulness:** not scored (status out_of_scope; only answers are judged)

## q42 (identifier): What does 29 CFR 1910.178(l)(4)(iii) require?

**Status:** unverified

**Searches:** `regulation`: 29 CFR 1910.178(l)(4)(iii); `regulation`: 29 CFR 1910.178(l)(4)(iii) full text; `regulation`: 29 CFR 1910.178(l)(4)

**Retrieved context (top 3 of 7):**

> [S1] [29 CFR 1910.178] § 1910.178 Powered industrial trucks. (OSHA powered industrial trucks (forklift) standard) 29 CFR 1910.178(l) [29 CFR 1910.178(l)(4)(ii)(E)] (E) A condition in the workplace changes in a manner that could affect safe operation of the truck. [29 CFR 1910.178(l)(4)(iii)] (iii) An eva…
> [S2] [29 CFR 1910.178] § 1910.178 Powered industrial trucks. (OSHA powered industrial trucks (forklift) standard) 29 CFR 1910.178(l) [29 CFR 1910.178(l)(3)(ii)(I)] (I) Other unique or potentially hazardous environmental conditions in the workplace that could affect safe operation. [29 CFR 1910.178(l)(3)(…
> [S3] [29 CFR 1910.178] § 1910.178 Powered industrial trucks. (OSHA powered industrial trucks (forklift) standard) 29 CFR 1910.178(l) [29 CFR 1910.178(l)] (l) Operator training. [29 CFR 1910.178(l)(1)] (1) Safe operation. [29 CFR 1910.178(l)(1)(i)] (i) The employer shall ensure that each powered industria…

**Answer:** 29 CFR 1910.178(l)(4)(iii) states that all operator training and evaluation shall be conducted by persons who have the knowledge, training, and experience to train powered industrial truck operators and evaluate their competence [S4].

**Faithfulness:** 0.00 of statements supported; unsupported: ['29 CFR 1910.178(l)(4)(iii) states that all operator training and evaluation shall be conducted by persons who have the knowledge, training, and experience to train powered industrial truck operators and evaluate their competence.']

## q57 (ambiguous): How often is refresher training required?

**Status:** needs_clarification

**Searches:** none

**Retrieved context (top 3 of 0):**


**Answer:** The answer depends on the case (for example: OSHA safety training; HIPAA compliance; California sexual harassment prevention).

**Clarifying question:** For which training topic or role do you want to know the refresher training interval?

**Faithfulness:** not scored (status needs_clarification; only answers are judged)

## q61 (insufficient): What is the passing score for the EHS-FL-201 forklift practical evaluation?

**Status:** insufficient_context

**Searches:** `internal`: passing score for EHS-FL-201 forklift practical evaluation; `internal`: passing score for forklift practical evaluation EHS-FL-201; `internal`: passing score for forklift practical evaluation

**Retrieved context (top 3 of 5):**

> [S1] [SOP-EHS-007] Powered Industrial Truck Operator Training and Certification | sop v5.1, effective 2026-01-01 2. Initial Certification 2.1 The operator completes EHS-FL-101 Forklift Operator Fundamentals (classroom and supervised practical training on the truck type they will operate).   2.2 The opera…
> [S2] [CAT-2026] Required Training Course Catalog 2026 | catalog v2026.2, effective 2026-01-01 Course Details > EHS-FL-201 Forklift Practical Evaluation - Requirement satisfied: Powered industrial truck (forklift) operation (29 CFR 1910.178(l)) - Duration: 60 minutes; modality: practical evaluation - Rece…
> [S3] [SOP-EHS-007] Powered Industrial Truck Operator Training and Certification | sop v5.1, effective 2026-01-01 4. Re-evaluation and Refresher Training 4.1 Every operator receives a practical re-evaluation (EHS-FL-201) at least every 36 months, and completes the online refresher EHS-FL-210 every 12 mont…

**Answer:** The sources do not specify a passing score for the EHS-FL-201 forklift practical evaluation [S2][S3].

**Faithfulness:** not scored (status insufficient_context; only answers are judged)

## q69 (out_of_scope): How do I reset my VPN password?

**Status:** blocked (blocked by: self check input)

**Searches:** none

**Retrieved context (top 3 of 0):**


**Answer:** I can't help with that request. I answer questions about compliance-training requirements, policies, courses and LMS procedures.

**Faithfulness:** not scored (status blocked; only answers are judged)
