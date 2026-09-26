---
doc_id: SOP-LMS-021
title: Overdue Training Escalation
doc_type: sop
version: "2.3"
effective_date: 2026-01-01
status: current
owner: Learning Systems
---

# SOP-LMS-021 Overdue Training Escalation

## 1. Purpose

Defines the automated reminders and escalations for required training that is approaching or past its due date.

## 2. Escalation Schedule

Days are counted relative to the due date (negative numbers are before the due date).

| Day | Action |
|---|---|
| -14 | reminder email to learner |
| -3 | second reminder to learner |
| 0 | status set to Overdue; manager notified |
| 7 | HR business partner notified; item appears on site compliance dashboard |
| 15 | site leader notified; end of grace period for annual non-safety-critical items |
| 30 | Compliance Office notified; may be recorded in performance review |

## 3. Safety-Critical Items

3.1 Safety-critical items (see POL-CT-001 3.2) have no grace period. On day 0 the LMS sends an integration event that suspends the related authorization: the forklift badge is deactivated in the access control system for forklift items, and the confined space roster entry is suspended for confined space items.

3.2 For bloodborne pathogens, the clinic manager must reassign the employee to duties without occupational exposure until EHS-BBP-100 is completed.

3.3 Authorizations are re-activated automatically within one hour of the completion being recorded.

## 4. Exceptions

4.1 Employees on approved leave of more than 14 days have their due dates paused; the item is due 30 days after return, or before the task if safety-critical.

4.2 Managers cannot extend due dates. Extensions of up to 30 days for non-safety-critical items may be granted by the Compliance Office and are recorded with a reason code.

## 5. Reporting

5.1 The site compliance dashboard shows completion rate, overdue count by days overdue (1-15, 16-30, 31-60, more than 60), and suspended authorizations.

5.2 Completion below 95% for any required course at a site for two consecutive months is reported to the Chief Compliance Officer.
