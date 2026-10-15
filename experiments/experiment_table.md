# Experiment table

Baseline run `20261015T152638Z-v0-c7c96975`, split `dev`. Every run is scored by the same versions: checks `checks-1-66c5a50a`, faithfulness `faith-v2-d130a39f`, relevance `rel-v2-c825e4f7`. Judge numbers come from an **uncalibrated judge** (faithfulness) and an **uncalibrated judge** (relevance).

Intervals are 95% Wilson intervals. The p-value is an exact two-sided sign test on the queries whose ungrounded flag flipped. Each run is one sample of a cloud model; the `seed43` row shows how much the numbers move with no real change. About 12 runs were compared on the same queries, so the best one looks better than it is; Step 12 confirms the chosen config once on held-out queries.

| Variable | Run | Ungrounded (n) | Change | Improved / regressed | p | Bad answer | False refusals | hit@4 | MRR | Latency | Failed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | `20261015T152638Z-v0-c7c96975` | 8.75% [4.30%, 16.98%] (7/80) | - | - | - | 10.00% [5.15%, 18.51%] (8/80) | 1 | 98.53% [92.13%, 99.74%] (67/68) | 0.944 | p50 831 / p95 2117 ms | 0 |
| generation.seed 42 -> 43 (noise control) | `20261015T215947Z-v1_seed43-cf6d47d3` | 6.25% [2.70%, 13.81%] (5/80) | -2.50 pts | 2 / 0 | 0.5000 | 7.50% [3.48%, 15.41%] (6/80) | 1 | 98.53% [92.13%, 99.74%] (67/68) | 0.944 | p50 809 / p95 1954 ms | 0 |
| generation.prompt_template: answer only from the context | `20261015T220338Z-v1_prompt_grounded-3cb0651e` | 6.25% [2.70%, 13.81%] (5/80) | -2.50 pts | 4 / 2 | 0.6875 | 7.50% [3.48%, 15.41%] (6/80) | 1 | 98.53% [92.13%, 99.74%] (67/68) | 0.944 | p50 744 / p95 1008 ms | 0 |
| generation.prompt_template + context_chunk_template: numbered chunks and [n] citations | `20261015T220912Z-v1_prompt_citations-23b6da7c` | 2.50% [0.69%, 8.66%] (2/80) | -6.25 pts | 5 / 0 | 0.0625 | 3.75% [1.28%, 10.45%] (3/80) | 1 | 98.53% [92.13%, 99.74%] (67/68) | 0.944 | p50 755 / p95 1385 ms | 0 |
| generation.prompt_template: explicit refusal sentence | `20261015T221540Z-v1_prompt_refusal-0a78ad5c` | 3.75% [1.28%, 10.45%] (3/80) | -5.00 pts | 4 / 0 | 0.1250 | 7.50% [3.48%, 15.41%] (6/80) | 3 | 98.53% [92.13%, 99.74%] (67/68) | 0.944 | p50 707 / p95 1037 ms | 0 |
| chunking.chunk_size 150 -> 75 | `20261015T221834Z-v1_chunk75-68312bcf` | 2.50% [0.69%, 8.66%] (2/80) | -6.25 pts | 5 / 0 | 0.0625 | 5.00% [1.96%, 12.16%] (4/80) | 2 | 98.53% [92.13%, 99.74%] (67/68) | 0.963 | p50 742 / p95 1072 ms | 0 |
| chunking.chunk_size 150 -> 300 | `20261015T222356Z-v1_chunk300-75ec7722` | 6.25% [2.70%, 13.81%] (5/80) | -2.50 pts | 4 / 2 | 0.6875 | 11.25% [6.03%, 20.02%] (9/80) | 4 | 92.65% [83.91%, 96.82%] (63/68) | 0.777 | p50 799 / p95 1925 ms | 0 |
| chunking.chunk_overlap 30 -> 0 | `20261015T222943Z-v1_overlap0-324866ad` | 11.25% [6.03%, 20.02%] (9/80) | +2.50 pts | 4 / 6 | 0.7539 | 16.25% [9.75%, 25.84%] (13/80) | 4 | 94.12% [85.83%, 97.69%] (64/68) | 0.902 | p50 762 / p95 1187 ms | 0 |
| retrieval.top_k 4 -> 2 | `20261015T223401Z-v1_topk2-3c74a564` | 5.00% [1.96%, 12.16%] (4/80) | -3.75 pts | 3 / 0 | 0.2500 | 12.50% [6.93%, 21.50%] (10/80) | 4 | - | 0.934 | p50 705 / p95 958 ms | 0 |
| retrieval.top_k 4 -> 6 | `20261015T223749Z-v1_topk6-217f7b9a` | 7.50% [3.48%, 15.41%] (6/80) | -1.25 pts | 4 / 3 | 1.0000 | 8.75% [4.30%, 16.98%] (7/80) | 1 | 98.53% [92.13%, 99.74%] (67/68) | 0.944 | p50 764 / p95 1092 ms | 0 |
| retrieval.method vector -> hybrid (BM25 + vector, RRF) | `20261015T224133Z-v1_hybrid-12a7c4ec` | 5.00% [1.96%, 12.16%] (4/80) | -3.75 pts | 4 / 1 | 0.3750 | 6.25% [2.70%, 13.81%] (5/80) | 1 | 98.53% [92.13%, 99.74%] (67/68) | 0.950 | p50 750 / p95 1023 ms | 0 |
| retrieval.rerank.enabled false -> true (top 10 -> top 4) | `20261015T224509Z-v1_rerank-d2d11546` | 3.75% [1.28%, 10.45%] (3/80) | -5.00 pts | 5 / 1 | 0.2188 | 6.25% [2.70%, 13.81%] (5/80) | 2 | 97.06% [89.90%, 99.19%] (66/68) | 0.949 | p50 850 / p95 1458 ms | 0 |
| query_rewrite.enabled false -> true | `20261015T224841Z-v1_rewrite-a4918ecb` | 1.25% [0.22%, 6.75%] (1/80) | -7.50 pts | 6 / 0 | 0.0312 | 6.25% [2.70%, 13.81%] (5/80) | 4 | 97.06% [89.90%, 99.19%] (66/68) | 0.923 | p50 1404 / p95 3552 ms | 0 |

## Flipped answers

### generation.seed 42 -> 43 (noise control) (`20261015T215947Z-v1_seed43-cf6d47d3`)

- Hypothesis (written before the run): Nothing real changes; only sampling noise can differ. Expect the ungrounded count within 2 of the baseline 7 and 0 to 3 flips in either direction. This run sets the noise floor for reading every other experiment.
- Result: improved 2, regressed 0; p = 0.5000, not statistically distinguishable from no change
- Mean faithfulness difference: +0.010 [-0.001, +0.027] (n=80)
- Changed keys: generation.context_chunk_template, generation.seed, query_rewrite.enabled, query_rewrite.prompt_template, retrieval.bm25.b, retrieval.bm25.k1, retrieval.hybrid.fusion, retrieval.hybrid.pool_k, retrieval.hybrid.rrf_k, retrieval.rerank.enabled, retrieval.rerank.model, retrieval.rerank.pool_k, retrieval.rerank.revision, retrieval.rerank.worker_timeout_s
- Not paired (a judge or output failed on either side): none

**q068** (multi_chunk): improved
- Question: On a domestic business trip where I receive per diem, can I also claim my restaurant bills?
- Baseline answer: No, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Experiment answer: No. When per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q082** (multi_chunk): improved
- Question: If I join on the 25th of a month, when do I get my first salary, and when is the payslip available?
- Baseline answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the payroll cut-off date of the 20th). Your payslip will be available on KestrelHub by the 5th of the following month.
- Experiment answer: Since you joined after the 20th of the month (the payroll cut-off date), you will receive your first month's salary together with the next month's payroll. Your payslip will be available on KestrelHub by the 5th of the following month.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8; contradicted: claim 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

### generation.prompt_template: answer only from the context (`20261015T220338Z-v1_prompt_grounded-3cb0651e`)

- Hypothesis (written before the run): All 7 baseline ungrounded answers are generation errors, so telling the model to use only the context should reduce answers that add outside facts. Expect ungrounded lower than baseline, with false refusals equal or slightly higher.
- Result: improved 4, regressed 2; p = 0.6875, not statistically distinguishable from no change
- Mean faithfulness difference: +0.013 [-0.013, +0.037] (n=80)
- Changed keys: generation.context_chunk_template, generation.prompt_template, query_rewrite.enabled, query_rewrite.prompt_template, retrieval.bm25.b, retrieval.bm25.k1, retrieval.hybrid.fusion, retrieval.hybrid.pool_k, retrieval.hybrid.rrf_k, retrieval.rerank.enabled, retrieval.rerank.model, retrieval.rerank.pool_k, retrieval.rerank.revision, retrieval.rerank.worker_timeout_s
- Not paired (a judge or output failed on either side): none

**q029** (lookup): improved
- Question: When does a full-time employee need a medical certificate for sick leave?
- Baseline answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave that exceeds 2 consecutive days.
- Experiment answer: A medical certificate from a registered medical practitioner is required for sick leave of more than 2 consecutive days.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q057** (paraphrased): improved
- Question: Is it okay to paste our source code into ChatGPT to debug it?
- Baseline answer: No. According to the policy on Generative AI tools, public AI tools may only be used with Public or Internal information; Confidential and Restricted information must never be entered into public AI tools. Kestrel Assist is the only approved AI tool for Confidential information.
- Experiment answer: The provided text does not mention ChatGPT specifically, but it states that public AI tools may be used only with Public or Internal information, and Confidential and Restricted information must never be entered into public AI tools.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 2, 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q058** (paraphrased): improved
- Question: My laptop won't let me install a program without admin rights. What should I do?
- Baseline answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk.
- Experiment answer: Temporary administrator rights for up to 24 hours can be requested through a KestrelDesk ticket.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q082** (multi_chunk): improved
- Question: If I join on the 25th of a month, when do I get my first salary, and when is the payslip available?
- Baseline answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the payroll cut-off date of the 20th). Your payslip will be available on KestrelHub by the 5th of the following month.
- Experiment answer: You will receive your first month's salary together with the next month's payroll, and the payslip will be published on KestrelHub by the 5th of the following month.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8; contradicted: claim 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q051** (paraphrased): regressed
- Question: As a trainee, can I take a day off in my very first month?
- Baseline answer: No, you cannot take casual leave in your first month because casual leave is credited only at the end of each completed month of internship.
- Experiment answer: An intern cannot take casual leave in the first month because it is credited at the end of each completed month.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.50 < 0.8

**q073** (multi_chunk): regressed
- Question: If I attend an external conference, how is my travel booked and what budget does it count against?
- Baseline answer: Travel for an external conference is booked under the Travel Policy, and the travel, stay, and conference fees count towards the annual learning budget.
- Experiment answer: Travel is booked under the Travel Policy and conference fees, travel and stay count towards the annual learning budget.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.75 < 0.8

### generation.prompt_template + context_chunk_template: numbered chunks and [n] citations (`20261015T220912Z-v1_prompt_citations-23b6da7c`)

- Hypothesis (written before the run): Having to cite a chunk for each fact pushes the model to state only what a chunk says. Expect ungrounded equal or lower; the [n] markers may change how the judge splits claims, and citation correctness is not scored.
- Result: improved 5, regressed 0; p = 0.0625, not statistically distinguishable from no change
- Mean faithfulness difference: +0.026 [+0.006, +0.050] (n=80)
- Changed keys: generation.context_chunk_template, generation.prompt_template, query_rewrite.enabled, query_rewrite.prompt_template, retrieval.bm25.b, retrieval.bm25.k1, retrieval.hybrid.fusion, retrieval.hybrid.pool_k, retrieval.hybrid.rrf_k, retrieval.rerank.enabled, retrieval.rerank.model, retrieval.rerank.pool_k, retrieval.rerank.revision, retrieval.rerank.worker_timeout_s
- Not paired (a judge or output failed on either side): none

**q029** (lookup): improved
- Question: When does a full-time employee need a medical certificate for sick leave?
- Baseline answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave that exceeds 2 consecutive days.
- Experiment answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave that exceeds 2 consecutive days [1].
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q058** (paraphrased): improved
- Question: My laptop won't let me install a program without admin rights. What should I do?
- Baseline answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk.
- Experiment answer: You can request temporary administrator rights for up to 24 hours by submitting a ticket through KestrelDesk [1].
- Judge reasoning, baseline: faithfulness 0.67 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q068** (multi_chunk): improved
- Question: On a domestic business trip where I receive per diem, can I also claim my restaurant bills?
- Baseline answer: No, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Experiment answer: No, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy [2].
- Judge reasoning, baseline: faithfulness 0.50 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q071** (multi_chunk): improved
- Question: An intern gets a pre-placement offer and joins full-time in band B4. How long is their probation, and what did they need to be considered for the PPO?
- Baseline answer: Their probation period is 3 months (as interns converting to full-time serve a 3-month probation regardless of band). To be considered for the PPO, they needed to complete at least 3 months of their internship and receive an "Exceeds expectations" rating in their final review.
- Experiment answer: An intern who receives a pre-placement offer and joins as a full-time employee serves a probation period of 3 months, regardless of their band [3]. To be considered for the PPO, the intern must have completed at least 3 months and received an "Exceeds expectations" rating in their final review [1].
- Judge reasoning, baseline: faithfulness 0.75 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q082** (multi_chunk): improved
- Question: If I join on the 25th of a month, when do I get my first salary, and when is the payslip available?
- Baseline answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the payroll cut-off date of the 20th). Your payslip will be available on KestrelHub by the 5th of the following month.
- Experiment answer: If you join after the 20th of the month, you will receive your first month's salary together with the next month's payroll [1]. Your payslip will be published on KestrelHub by the 5th of the following month [1], [4].
- Judge reasoning, baseline: faithfulness 0.67 < 0.8; contradicted: claim 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

### generation.prompt_template: explicit refusal sentence (`20261015T221540Z-v1_prompt_refusal-0a78ad5c`)

- Hypothesis (written before the run): An explicit refusal sentence should keep unanswerable questions refused (0 of 12 ungrounded at baseline) and may make the model refuse answerable ones. Expect ungrounded about equal and false refusals equal or higher.
- Result: improved 4, regressed 0; p = 0.1250, not statistically distinguishable from no change
- Mean faithfulness difference: +0.022 [+0.004, +0.044] (n=80)
- Changed keys: generation.context_chunk_template, generation.prompt_template, query_rewrite.enabled, query_rewrite.prompt_template, retrieval.bm25.b, retrieval.bm25.k1, retrieval.hybrid.fusion, retrieval.hybrid.pool_k, retrieval.hybrid.rrf_k, retrieval.rerank.enabled, retrieval.rerank.model, retrieval.rerank.pool_k, retrieval.rerank.revision, retrieval.rerank.worker_timeout_s
- Not paired (a judge or output failed on either side): none

**q029** (lookup): improved
- Question: When does a full-time employee need a medical certificate for sick leave?
- Baseline answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave that exceeds 2 consecutive days.
- Experiment answer: A medical certificate from a registered medical practitioner is required for sick leave of more than 2 consecutive days.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q057** (paraphrased): improved
- Question: Is it okay to paste our source code into ChatGPT to debug it?
- Baseline answer: No. According to the policy on Generative AI tools, public AI tools may only be used with Public or Internal information; Confidential and Restricted information must never be entered into public AI tools. Kestrel Assist is the only approved AI tool for Confidential information.
- Experiment answer: The context does not contain this information.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 2, 3
- Judge reasoning, experiment: no factual claims

**q058** (paraphrased): improved
- Question: My laptop won't let me install a program without admin rights. What should I do?
- Baseline answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk.
- Experiment answer: Temporary administrator rights for up to 24 hours can be requested through a KestrelDesk ticket.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q082** (multi_chunk): improved
- Question: If I join on the 25th of a month, when do I get my first salary, and when is the payslip available?
- Baseline answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the payroll cut-off date of the 20th). Your payslip will be available on KestrelHub by the 5th of the following month.
- Experiment answer: You will receive your first month's salary together with the next month's payroll, and the payslip will be available on KestrelHub by the 5th of the following month.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8; contradicted: claim 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

### chunking.chunk_size 150 -> 75 (`20261015T221834Z-v1_chunk75-68312bcf`)

- Hypothesis (written before the run): Smaller chunks give more focused context but can split a fact across chunks. Expect hit@4 equal or slightly lower, and multi_chunk questions to suffer most; ungrounded within noise or higher.
- Result: improved 5, regressed 0; p = 0.0625, not statistically distinguishable from no change
- Mean faithfulness difference: +0.026 [+0.006, +0.050] (n=80)
- Changed keys: chunking.chunk_size, generation.context_chunk_template, query_rewrite.enabled, query_rewrite.prompt_template, retrieval.bm25.b, retrieval.bm25.k1, retrieval.hybrid.fusion, retrieval.hybrid.pool_k, retrieval.hybrid.rrf_k, retrieval.rerank.enabled, retrieval.rerank.model, retrieval.rerank.pool_k, retrieval.rerank.revision, retrieval.rerank.worker_timeout_s
- Not paired (a judge or output failed on either side): none

**q029** (lookup): improved
- Question: When does a full-time employee need a medical certificate for sick leave?
- Baseline answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave that exceeds 2 consecutive days.
- Experiment answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave of more than 2 consecutive days.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q058** (paraphrased): improved
- Question: My laptop won't let me install a program without admin rights. What should I do?
- Baseline answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk.
- Experiment answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a KestrelDesk ticket.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q068** (multi_chunk): improved
- Question: On a domestic business trip where I receive per diem, can I also claim my restaurant bills?
- Baseline answer: No, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Experiment answer: No, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q071** (multi_chunk): improved
- Question: An intern gets a pre-placement offer and joins full-time in band B4. How long is their probation, and what did they need to be considered for the PPO?
- Baseline answer: Their probation period is 3 months (as interns converting to full-time serve a 3-month probation regardless of band). To be considered for the PPO, they needed to complete at least 3 months of their internship and receive an "Exceeds expectations" rating in their final review.
- Experiment answer: Their probation period is 3 months. To be considered for the PPO, they needed to complete at least 3 months of their internship and receive an "Exceeds expectations" rating in their final review.
- Judge reasoning, baseline: faithfulness 0.75 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q082** (multi_chunk): improved
- Question: If I join on the 25th of a month, when do I get my first salary, and when is the payslip available?
- Baseline answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the payroll cut-off date of the 20th). Your payslip will be available on KestrelHub by the 5th of the following month.
- Experiment answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll. Your payslip will be available on KestrelHub by the 5th of the following month.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8; contradicted: claim 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

### chunking.chunk_size 150 -> 300 (`20261015T222356Z-v1_chunk300-75ec7722`)

- Hypothesis (written before the run): Larger chunks keep related facts of one policy together, so multi_chunk questions need fewer chunks, but each chunk carries more unrelated text. Expect hit@4 equal or higher; ungrounded within noise.
- Result: improved 4, regressed 2; p = 0.6875, not statistically distinguishable from no change
- Mean faithfulness difference: +0.003 [-0.033, +0.033] (n=80)
- Changed keys: chunking.chunk_size, generation.context_chunk_template, query_rewrite.enabled, query_rewrite.prompt_template, retrieval.bm25.b, retrieval.bm25.k1, retrieval.hybrid.fusion, retrieval.hybrid.pool_k, retrieval.hybrid.rrf_k, retrieval.rerank.enabled, retrieval.rerank.model, retrieval.rerank.pool_k, retrieval.rerank.revision, retrieval.rerank.worker_timeout_s
- Not paired (a judge or output failed on either side): none

**q057** (paraphrased): improved
- Question: Is it okay to paste our source code into ChatGPT to debug it?
- Baseline answer: No. According to the policy on Generative AI tools, public AI tools may only be used with Public or Internal information; Confidential and Restricted information must never be entered into public AI tools. Kestrel Assist is the only approved AI tool for Confidential information.
- Experiment answer: No, it is not okay. According to the Information Security Policy, source code is classified as **Confidential** information. The policy states that Confidential and Restricted information must never be entered into public AI tools; **Kestrel Assist** is the only approved AI tool for Confidential information.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 2, 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q058** (paraphrased): improved
- Question: My laptop won't let me install a program without admin rights. What should I do?
- Baseline answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk.
- Experiment answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk, the IT service desk.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q068** (multi_chunk): improved
- Question: On a domestic business trip where I receive per diem, can I also claim my restaurant bills?
- Baseline answer: No, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Experiment answer: No, you cannot. According to the Business Travel Policy (Section 4), when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q071** (multi_chunk): improved
- Question: An intern gets a pre-placement offer and joins full-time in band B4. How long is their probation, and what did they need to be considered for the PPO?
- Baseline answer: Their probation period is 3 months (as interns converting to full-time serve a 3-month probation regardless of band). To be considered for the PPO, they needed to complete at least 3 months of their internship and receive an "Exceeds expectations" rating in their final review.
- Experiment answer: Their probation period is 3 months. To be considered for the PPO, the intern needed to complete at least 3 months of their internship and receive an "Exceeds expectations" rating in their final review.
- Judge reasoning, baseline: faithfulness 0.75 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q076** (multi_chunk): regressed
- Question: What disciplinary steps can follow a violation of the Information Security Policy?
- Baseline answer: Violations of the Information Security Policy are handled under the disciplinary process in the Code of Conduct, which consists of the following stages:
- Verbal warning
- Written warning
- Final written warning
- Termination

However, serious misconduct can lead to immediate termination without going through the earlier stages.
- Experiment answer: Violations of the Information Security Policy are handled under the disciplinary process in the Code of Conduct.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.00 < 0.8

**q092** (unanswerable): regressed
- Question: Does Kestrel Labs reimburse a gym membership?
- Baseline answer: Based on the provided text, there is no mention of gym memberships; therefore, the documents do not specify whether Kestrel Labs reimburses them.
- Experiment answer: No, Kestrel Labs does not reimburse gym memberships. According to the Expense Reimbursement Policy, "personal grooming or personal shopping" and "personal entertainment" are listed as not reimbursable.
- Judge reasoning, baseline: no factual claims
- Judge reasoning, experiment: faithfulness 0.67 < 0.8

### chunking.chunk_overlap 30 -> 0 (`20261015T222943Z-v1_overlap0-324866ad`)

- Hypothesis (written before the run): Without overlap a fact on a chunk boundary is cut in two. Expect hit@4 unchanged or one query lower, and ungrounded within noise.
- Result: improved 4, regressed 6; p = 0.7539, not statistically distinguishable from no change
- Mean faithfulness difference: -0.017 [-0.059, +0.020] (n=80)
- Changed keys: chunking.chunk_overlap, generation.context_chunk_template, query_rewrite.enabled, query_rewrite.prompt_template, retrieval.bm25.b, retrieval.bm25.k1, retrieval.hybrid.fusion, retrieval.hybrid.pool_k, retrieval.hybrid.rrf_k, retrieval.rerank.enabled, retrieval.rerank.model, retrieval.rerank.pool_k, retrieval.rerank.revision, retrieval.rerank.worker_timeout_s
- Not paired (a judge or output failed on either side): none

**q029** (lookup): improved
- Question: When does a full-time employee need a medical certificate for sick leave?
- Baseline answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave that exceeds 2 consecutive days.
- Experiment answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave of more than 2 consecutive days.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q058** (paraphrased): improved
- Question: My laptop won't let me install a program without admin rights. What should I do?
- Baseline answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk.
- Experiment answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a KestrelDesk ticket.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q068** (multi_chunk): improved
- Question: On a domestic business trip where I receive per diem, can I also claim my restaurant bills?
- Baseline answer: No, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Experiment answer: No. According to the Per diem section, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q082** (multi_chunk): improved
- Question: If I join on the 25th of a month, when do I get my first salary, and when is the payslip available?
- Baseline answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the payroll cut-off date of the 20th). Your payslip will be available on KestrelHub by the 5th of the following month.
- Experiment answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the 20th cut-off date). Your payslip will be available on KestrelHub by the 5th of the following month.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8; contradicted: claim 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q010** (lookup): regressed
- Question: Who has to approve an expense claim of ₹30,000?
- Baseline answer: An expense claim of ₹30,000 requires approval from both the reporting manager and the department head.
- Experiment answer: An expense claim of ₹30,000 requires approval from both the reporting manager and the department head.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.50 < 0.8

**q051** (paraphrased): regressed
- Question: As a trainee, can I take a day off in my very first month?
- Baseline answer: No, you cannot take casual leave in your first month because casual leave is credited only at the end of each completed month of internship.
- Experiment answer: No. According to the Intern Leave Policy, casual leave is credited at the end of each completed month, so an intern cannot take casual leave in the first month.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.67 < 0.8

**q054** (paraphrased): regressed
- Question: Can I check my work emails on my own phone?
- Baseline answer: Yes, you may use your personal mobile phone for company email only after it has been enrolled in the company mobile device management (MDM) app. However, if you are an intern, you may not use any personal device for company work.
- Experiment answer: No. According to the policy, personal tablets and phones must not be used to access any company system, and interns specifically may not use any personal device for company work. Company systems must be accessed only from the company laptop and over the company VPN.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.00 < 0.8

**q056** (paraphrased): regressed
- Question: I'm based in Pune. Do we get the first of May off?
- Baseline answer: Yes, you do. Maharashtra Day is a location-specific fixed holiday for the Pune office on Friday, 1 May 2026.
- Experiment answer: Yes, you do. The Pune office has a fixed holiday for Maharashtra Day on Friday, 1 May 2026.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.50 < 0.8

**q072** (multi_chunk): regressed
- Question: After I resign, can I use my earned leave to leave earlier, and how much of it will be paid out?
- Baseline answer: No, you cannot use earned leave to shorten your notice period. Regarding payment, earned leave is encashable at exit up to a maximum of 30 days, calculated based on your last drawn basic salary.
- Experiment answer: No, you cannot use earned leave to shorten your notice period. Regarding payment, earned leave is encashable only at exit up to a maximum of 30 days, calculated based on your last drawn basic salary.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.67 < 0.8

**q078** (multi_chunk): regressed
- Question: What does the 90-day check-in feed into for a new hire in band B5, and when is that review done?
- Baseline answer: The 90-day check-in feeds into the probation review. For an employee in band B5, the probation period is 3 months, and the review is completed by the reporting manager on KestrelHub in the last month of probation (month 3).
- Experiment answer: The 90-day check-in feeds into the probation review. For a new hire in band B5, this review is completed by the reporting manager in the third month of probation.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.67 < 0.8

### retrieval.top_k 4 -> 2 (`20261015T223401Z-v1_topk2-3c74a564`)

- Hypothesis (written before the run): Fewer chunks lose facts that sit in a third or fourth chunk. Expect hit@4-style recall and multi_chunk answers to get worse (more unsupported or refused answers); less context may slightly reduce unsupported extras for lookups.
- Result: improved 3, regressed 0; p = 0.2500, not statistically distinguishable from no change
- Mean faithfulness difference: +0.018 [+0.003, +0.038] (n=80)
- Changed keys: generation.context_chunk_template, query_rewrite.enabled, query_rewrite.prompt_template, retrieval.bm25.b, retrieval.bm25.k1, retrieval.hybrid.fusion, retrieval.hybrid.pool_k, retrieval.hybrid.rrf_k, retrieval.rerank.enabled, retrieval.rerank.model, retrieval.rerank.pool_k, retrieval.rerank.revision, retrieval.rerank.worker_timeout_s, retrieval.top_k
- Not paired (a judge or output failed on either side): none

**q029** (lookup): improved
- Question: When does a full-time employee need a medical certificate for sick leave?
- Baseline answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave that exceeds 2 consecutive days.
- Experiment answer: A medical certificate from a registered medical practitioner is required for sick leave of more than 2 consecutive days.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q058** (paraphrased): improved
- Question: My laptop won't let me install a program without admin rights. What should I do?
- Baseline answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk.
- Experiment answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk, the IT service desk.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q082** (multi_chunk): improved
- Question: If I join on the 25th of a month, when do I get my first salary, and when is the payslip available?
- Baseline answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the payroll cut-off date of the 20th). Your payslip will be available on KestrelHub by the 5th of the following month.
- Experiment answer: You will receive your first month's salary together with the next month's payroll, and the payslip will be available on KestrelHub by the 5th of the following month.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8; contradicted: claim 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

### retrieval.top_k 4 -> 6 (`20261015T223749Z-v1_topk6-217f7b9a`)

- Hypothesis (written before the run): More chunks raise recall, but add text from confusable policies (employee vs intern leave). Expect recall equal or higher and ungrounded within noise or slightly higher.
- Result: improved 4, regressed 3; p = 1.0000, not statistically distinguishable from no change
- Mean faithfulness difference: -0.002 [-0.040, +0.031] (n=80)
- Changed keys: generation.context_chunk_template, query_rewrite.enabled, query_rewrite.prompt_template, retrieval.bm25.b, retrieval.bm25.k1, retrieval.hybrid.fusion, retrieval.hybrid.pool_k, retrieval.hybrid.rrf_k, retrieval.rerank.enabled, retrieval.rerank.model, retrieval.rerank.pool_k, retrieval.rerank.revision, retrieval.rerank.worker_timeout_s, retrieval.top_k
- Not paired (a judge or output failed on either side): none

**q029** (lookup): improved
- Question: When does a full-time employee need a medical certificate for sick leave?
- Baseline answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave that exceeds 2 consecutive days.
- Experiment answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave of more than 2 consecutive days.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q058** (paraphrased): improved
- Question: My laptop won't let me install a program without admin rights. What should I do?
- Baseline answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk.
- Experiment answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk, the IT service desk.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q068** (multi_chunk): improved
- Question: On a domestic business trip where I receive per diem, can I also claim my restaurant bills?
- Baseline answer: No, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Experiment answer: No, you cannot. According to the Per diem section, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q082** (multi_chunk): improved
- Question: If I join on the 25th of a month, when do I get my first salary, and when is the payslip available?
- Baseline answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the payroll cut-off date of the 20th). Your payslip will be available on KestrelHub by the 5th of the following month.
- Experiment answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the 20th payroll cut-off date). Your payslip will be available on KestrelHub by the 5th of the following month.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8; contradicted: claim 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q051** (paraphrased): regressed
- Question: As a trainee, can I take a day off in my very first month?
- Baseline answer: No, you cannot take casual leave in your first month because casual leave is credited only at the end of each completed month of internship.
- Experiment answer: No, you cannot. According to the Intern Leave Policy, casual leave is credited at the end of each completed month, so an intern cannot take casual leave in the first month.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.67 < 0.8

**q076** (multi_chunk): regressed
- Question: What disciplinary steps can follow a violation of the Information Security Policy?
- Baseline answer: Violations of the Information Security Policy are handled under the disciplinary process in the Code of Conduct, which consists of the following stages:
- Verbal warning
- Written warning
- Final written warning
- Termination

However, serious misconduct can lead to immediate termination without going through the earlier stages.
- Experiment answer: Violations of the Information Security Policy are handled under the disciplinary process in the Code of Conduct, which consists of the following stages:
*   Verbal warning
*   Written warning
*   Final written warning
*   Termination

However, serious misconduct (such as harassment, violence, theft, or fraud) can lead to immediate termination without going through the earlier stages.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.56 < 0.8

**q099** (unanswerable): regressed
- Question: Can I claim reimbursement for an external monitor for my home office setup?
- Baseline answer: Based on the provided text, there is no mention of reimbursement for an external monitor. The only home setup support mentioned is a one-time reimbursement of up to ₹8,000 for an ergonomic chair for confirmed employees and a monthly internet claim of up to ₹1,000 for those on hybrid or remote work.
- Experiment answer: Based on the provided text, there is no mention of reimbursement for an external monitor. The only home setup support mentioned is a one-time reimbursement of up to ₹8,000 for an ergonomic chair and a monthly claim of up to ₹1,000 for home internet for employees on hybrid or remote work.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.00 < 0.8; contradicted: claim 1, 2

### retrieval.method vector -> hybrid (BM25 + vector, RRF) (`20261015T224133Z-v1_hybrid-12a7c4ec`)

- Hypothesis (written before the run): BM25 adds exact-word matches (numbers, band names, system names) to vector search. Expect MRR equal or slightly higher. Retrieval is near ceiling (one miss at baseline), so ungrounded can change by at most about one answer.
- Result: improved 4, regressed 1; p = 0.3750, not statistically distinguishable from no change
- Mean faithfulness difference: +0.016 [-0.007, +0.039] (n=80)
- Changed keys: generation.context_chunk_template, query_rewrite.enabled, query_rewrite.prompt_template, retrieval.bm25.b, retrieval.bm25.k1, retrieval.hybrid.fusion, retrieval.hybrid.pool_k, retrieval.hybrid.rrf_k, retrieval.method, retrieval.rerank.enabled, retrieval.rerank.model, retrieval.rerank.pool_k, retrieval.rerank.revision, retrieval.rerank.worker_timeout_s
- Not paired (a judge or output failed on either side): none

**q029** (lookup): improved
- Question: When does a full-time employee need a medical certificate for sick leave?
- Baseline answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave that exceeds 2 consecutive days.
- Experiment answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave of more than 2 consecutive days.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q057** (paraphrased): improved
- Question: Is it okay to paste our source code into ChatGPT to debug it?
- Baseline answer: No. According to the policy on Generative AI tools, public AI tools may only be used with Public or Internal information; Confidential and Restricted information must never be entered into public AI tools. Kestrel Assist is the only approved AI tool for Confidential information.
- Experiment answer: No. According to the Information Security Policy, source code is classified as **Confidential** business information, and the policy states that company information must not be sent to personal email accounts or stored on personal cloud drives.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 2, 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q071** (multi_chunk): improved
- Question: An intern gets a pre-placement offer and joins full-time in band B4. How long is their probation, and what did they need to be considered for the PPO?
- Baseline answer: Their probation period is 3 months (as interns converting to full-time serve a 3-month probation regardless of band). To be considered for the PPO, they needed to complete at least 3 months of their internship and receive an "Exceeds expectations" rating in their final review.
- Experiment answer: Their probation period is 3 months. To be considered for the PPO, the intern needed to complete at least 3 months of their internship and receive an "Exceeds expectations" rating in their final review.
- Judge reasoning, baseline: faithfulness 0.75 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q082** (multi_chunk): improved
- Question: If I join on the 25th of a month, when do I get my first salary, and when is the payslip available?
- Baseline answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the payroll cut-off date of the 20th). Your payslip will be available on KestrelHub by the 5th of the following month.
- Experiment answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the 20th payroll cut-off). Your payslip will be available on KestrelHub by the 5th of the following month.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8; contradicted: claim 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q051** (paraphrased): regressed
- Question: As a trainee, can I take a day off in my very first month?
- Baseline answer: No, you cannot take casual leave in your first month because casual leave is credited only at the end of each completed month of internship.
- Experiment answer: No, you cannot take casual leave in your first month because casual leave is credited only at the end of each completed month.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.50 < 0.8

### retrieval.rerank.enabled false -> true (top 10 -> top 4) (`20261015T224509Z-v1_rerank-d2d11546`)

- Hypothesis (written before the run): The cross-encoder reorders the top 10 candidates so the best 4 enter the prompt. Expect MRR equal or slightly higher and ungrounded within noise, because retrieval is near ceiling. Latency rises.
- Result: improved 5, regressed 1; p = 0.2188, not statistically distinguishable from no change
- Mean faithfulness difference: +0.020 [-0.005, +0.046] (n=80)
- Changed keys: generation.context_chunk_template, query_rewrite.enabled, query_rewrite.prompt_template, retrieval.bm25.b, retrieval.bm25.k1, retrieval.hybrid.fusion, retrieval.hybrid.pool_k, retrieval.hybrid.rrf_k, retrieval.rerank.enabled, retrieval.rerank.model, retrieval.rerank.pool_k, retrieval.rerank.revision, retrieval.rerank.worker_timeout_s
- Not paired (a judge or output failed on either side): none

**q029** (lookup): improved
- Question: When does a full-time employee need a medical certificate for sick leave?
- Baseline answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave that exceeds 2 consecutive days.
- Experiment answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave of more than 2 consecutive days.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q058** (paraphrased): improved
- Question: My laptop won't let me install a program without admin rights. What should I do?
- Baseline answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk.
- Experiment answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk, the IT service desk.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q068** (multi_chunk): improved
- Question: On a domestic business trip where I receive per diem, can I also claim my restaurant bills?
- Baseline answer: No, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Experiment answer: No, you cannot. According to the Per diem section, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q071** (multi_chunk): improved
- Question: An intern gets a pre-placement offer and joins full-time in band B4. How long is their probation, and what did they need to be considered for the PPO?
- Baseline answer: Their probation period is 3 months (as interns converting to full-time serve a 3-month probation regardless of band). To be considered for the PPO, they needed to complete at least 3 months of their internship and receive an "Exceeds expectations" rating in their final review.
- Experiment answer: Their probation period is 3 months. To be considered for the PPO, the intern needed to complete at least 3 months and receive an "Exceeds expectations" rating in their final review.
- Judge reasoning, baseline: faithfulness 0.75 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q082** (multi_chunk): improved
- Question: If I join on the 25th of a month, when do I get my first salary, and when is the payslip available?
- Baseline answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the payroll cut-off date of the 20th). Your payslip will be available on KestrelHub by the 5th of the following month.
- Experiment answer: Since you joined after the 20th (the payroll cut-off date), you will receive your first month's salary together with the next month's payroll. Your payslip will be available on KestrelHub by the 5th of the following month.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8; contradicted: claim 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q051** (paraphrased): regressed
- Question: As a trainee, can I take a day off in my very first month?
- Baseline answer: No, you cannot take casual leave in your first month because casual leave is credited only at the end of each completed month of internship.
- Experiment answer: No, you cannot take casual leave in your first month because casual leave is credited only at the end of each completed month.
- Judge reasoning, baseline: faithfulness 1.00 >= 0.8, none contradicted
- Judge reasoning, experiment: faithfulness 0.50 < 0.8

### query_rewrite.enabled false -> true (`20261015T224841Z-v1_rewrite-a4918ecb`)

- Hypothesis (written before the run): Rewriting the question into a search query helps vague and paraphrased questions find the right document, but can drop exact words in lookups. Expect paraphrased hit@4 equal or higher and ungrounded within noise.
- Result: improved 6, regressed 0; p = 0.0312, significant at 0.05
- Mean faithfulness difference: +0.030 [+0.008, +0.056] (n=80)
- Changed keys: generation.context_chunk_template, query_rewrite.enabled, query_rewrite.prompt_template, retrieval.bm25.b, retrieval.bm25.k1, retrieval.hybrid.fusion, retrieval.hybrid.pool_k, retrieval.hybrid.rrf_k, retrieval.rerank.enabled, retrieval.rerank.model, retrieval.rerank.pool_k, retrieval.rerank.revision, retrieval.rerank.worker_timeout_s
- Not paired (a judge or output failed on either side): none

**q029** (lookup): improved
- Question: When does a full-time employee need a medical certificate for sick leave?
- Baseline answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave that exceeds 2 consecutive days.
- Experiment answer: A full-time employee is required to provide a medical certificate from a registered medical practitioner for sick leave of more than 2 consecutive days.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q057** (paraphrased): improved
- Question: Is it okay to paste our source code into ChatGPT to debug it?
- Baseline answer: No. According to the policy on Generative AI tools, public AI tools may only be used with Public or Internal information; Confidential and Restricted information must never be entered into public AI tools. Kestrel Assist is the only approved AI tool for Confidential information.
- Experiment answer: No, it is not okay. According to the Information Security Policy, source code is classified as "Confidential" information, and the policy explicitly states that "Confidential and Restricted information must never be entered into public AI tools." Kestrel Assist is the only approved AI tool for Confidential information.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8; contradicted: claim 2, 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q058** (paraphrased): improved
- Question: My laptop won't let me install a program without admin rights. What should I do?
- Baseline answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk.
- Experiment answer: You should request temporary administrator rights (valid for up to 24 hours) by submitting a ticket through KestrelDesk, the IT service desk.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q068** (multi_chunk): improved
- Question: On a domestic business trip where I receive per diem, can I also claim my restaurant bills?
- Baseline answer: No, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Experiment answer: No, you cannot. According to the Per diem section, when per diem is paid, meal bills cannot be claimed separately under the Expense Reimbursement Policy.
- Judge reasoning, baseline: faithfulness 0.50 < 0.8
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q071** (multi_chunk): improved
- Question: An intern gets a pre-placement offer and joins full-time in band B4. How long is their probation, and what did they need to be considered for the PPO?
- Baseline answer: Their probation period is 3 months (as interns converting to full-time serve a 3-month probation regardless of band). To be considered for the PPO, they needed to complete at least 3 months of their internship and receive an "Exceeds expectations" rating in their final review.
- Experiment answer: Their probation period is 3 months. To be considered for the PPO, the intern needed to complete at least 3 months of their internship and receive an "Exceeds expectations" rating in their final review.
- Judge reasoning, baseline: faithfulness 0.75 < 0.8; contradicted: claim 1
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted

**q082** (multi_chunk): improved
- Question: If I join on the 25th of a month, when do I get my first salary, and when is the payslip available?
- Baseline answer: If you join on the 25th of the month, you will receive your first month's salary together with the next month's payroll (as you joined after the payroll cut-off date of the 20th). Your payslip will be available on KestrelHub by the 5th of the following month.
- Experiment answer: Since you joined after the 20th of the month (the payroll cut-off date), you will receive your first month's salary together with the next month's payroll. Your payslip will be available on KestrelHub by the 5th of the following month.
- Judge reasoning, baseline: faithfulness 0.67 < 0.8; contradicted: claim 3
- Judge reasoning, experiment: faithfulness 1.00 >= 0.8, none contradicted
