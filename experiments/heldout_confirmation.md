# Held-out confirmation (Step 12)

Chosen config `configs/v1_prompt_citations.yaml` against `configs/v0.yaml`, each run once on the 20 held-out queries. Judge numbers come from an **uncalibrated judge** (faithfulness) and an **uncalibrated judge** (relevance).

## Verdict: `not_distinguishable`

not distinguishable from v0 on the held-out queries; 20 queries cannot detect a change of this size.

Improved 1, regressed 0, exact paired p = 1.0000 (pre-registered level 0.05).

## Ungrounded rate

Intervals are 95% Wilson intervals. One answer is 5 percentage points.

| Config | Held-out (unseen) | Dev | Held-out rate inside dev interval |
|---|---|---|---|
| v0 `20261015T233336Z-v0-9ff3bd9f` | 10.00% [2.79%, 30.10%] (2/20) | 8.75% [4.30%, 16.98%] (7/80) | True |
| chosen `20261015T233407Z-v1_prompt_citations-84777159` | 5.00% [0.89%, 23.61%] (1/20) | 2.50% [0.69%, 8.66%] (2/80) | True |

Mean faithfulness difference (chosen - v0): +0.028 [+0.000, +0.083] (n=20).

## Other held-out numbers

| Metric | v0 | chosen |
|---|---|---|
| Bad answer | 15.00% [5.24%, 36.04%] (3/20) | 10.00% [2.79%, 30.10%] (2/20) |
| Irrelevant | 5.00% [0.89%, 23.61%] (1/20) | 5.00% [0.89%, 23.61%] (1/20) |
| False refusals | 1 | 1 |
| hit@4 (answerable) | 100.00% [81.57%, 100.00%] (17/17) | 100.00% [81.57%, 100.00%] (17/17) |
| Failed outputs | 0 | 0 |

## Registration

- Selection rule: Lowest ungrounded rate among Step 11 dev runs whose bad-answer rate and false-refusal count are not worse than baseline, noise control excluded (fixed before the Step 11 results).
- Scorer versions: checks `checks-1-66c5a50a`, faithfulness `faith-v2-d130a39f`, relevance `rel-v2-c825e4f7`
- Config SHA-256: `configs/v0.yaml` `47c3955a26fea7aafc663e936b8c5501d8edae86bb2703ab0442d0b055f2de89`, `configs/v1_prompt_citations.yaml` `fa04b3e056bc990b6b0aaa828c42fba2b751abc38be205de143546b19f6aaff0`
- Amendments: 0

## Limitations

- n = 20 held-out queries: one answer is 5 percentage points, so this set cannot show a change of a few points.
- Per-type n is 3-10: no per-type claims are made.
- Judge numbers come from an uncalibrated judge (faithfulness) and an uncalibrated judge (relevance): they are the judge's measurement, not verified rates.
- The chosen config was the best of 12 dev looks, so its dev rate is optimistic (winner's curse); a held-out rate nearer the baseline is the normal case.
- Each answer is one sample from a cloud model that is not fully deterministic.
- CV hypothesis 25% -> 12% is a hypothesis, not a target; the CV follows the measurement.

## Answers to read

Every ungrounded answer and false refusal, for a human check (a human-review note, not part of any calibration statistic).

**q079** (multi_chunk), v0, ungrounded
- Question: Which types of employee leave can be carried forward, and up to what limits?
- Answer: Only sick leave (SL) can be carried forward, up to a maximum accumulated balance of 30 days.
- Retrieved chunks: leave_policy#001, leave_policy#003, resignation_and_exit_policy#001, leave_policy#000
- Judge reasoning: faithfulness 0.50 < 0.8

**q084** (multi_chunk), v0, ungrounded
- Question: How many paid holidays on weekdays does a Bengaluru employee get in 2026, counting fixed and optional holidays?
- Answer: A Bengaluru employee gets a total of **13** paid holidays on weekdays in 2026.

**Breakdown:**
*   **Common Fixed Holidays:** 9 (Out of the 10 common holidays, Independence Day falls on a Saturday and is not substituted).
*   **Location-Specific Fixed Holidays:** 1 (Diwali on Tuesday, 10 November; Karnataka Rajyotsava falls on a Sunday and is not substituted).
*   **Optional Holidays:** 3 (The employee can choose 2 optional holidays from the list, all of which fall on weekdays).

**Calculation:** 9 (Common) + 1 (Location-specific) + 3 (Optional) = **13** (Wait, the prompt asks for the total they *get* based on the policy).

**Correction:**
*   **Fixed Weekday Holidays:** 10 (9 common + 1 location-specific).
*   **Optional Weekday Holidays:** 2 (Employees may choose 2).
*   **Total:** **12** paid holidays on weekdays.
- Retrieved chunks: holiday_calendar_2026#000, holiday_calendar_2026#001, holiday_calendar_2026#002, remote_hybrid_work_policy#001
- Judge reasoning: faithfulness 0.70 < 0.8; contradicted: claim 5, 7

**q060** (paraphrased), v0, false refusal
- Question: Can I borrow money from the company against my pay?
- Answer: Based on the provided context, there is no information regarding borrowing money from the company against pay.
- Retrieved chunks: parental_leave_policy#001, payroll_policy#003, remote_hybrid_work_policy#001, leave_policy#003
- Judge reasoning: no factual claims

**q084** (multi_chunk), chosen config, ungrounded
- Question: How many paid holidays on weekdays does a Bengaluru employee get in 2026, counting fixed and optional holidays?
- Answer: A Bengaluru employee gets a total of 13 paid holidays on weekdays in 2026:

*   **Fixed Holidays:** There are 11 weekday fixed holidays. While there are 12 fixed holidays in total [1], Independence Day (Saturday, 15 August) and Karnataka Rajyotsava (Sunday, 1 November) fall on weekends and do not receive substitute holidays [2].
*   **Optional Holidays:** Employees may choose 2 optional holidays from the provided list [2]. All listed optional holidays (Makar Sankranti, Eid ul-Fitr, Eid ul-Adha, Raksha Bandhan, and Guru Nanak Jayanti) fall on weekdays [3].
- Retrieved chunks: holiday_calendar_2026#000, holiday_calendar_2026#001, holiday_calendar_2026#002, remote_hybrid_work_policy#001
- Judge reasoning: faithfulness 0.75 < 0.8; contradicted: claim 1, 2

**q060** (paraphrased), chosen config, false refusal
- Question: Can I borrow money from the company against my pay?
- Answer: Based on the provided text, there is no information regarding whether an employee can borrow money from the company against their pay.
- Retrieved chunks: parental_leave_policy#001, payroll_policy#003, remote_hybrid_work_policy#001, leave_policy#003
- Judge reasoning: no factual claims
