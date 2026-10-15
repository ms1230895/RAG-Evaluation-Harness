# Experiment log

One entry per experiment. Change ONE variable per experiment.

| Date | Config | Variable changed | Run ID | Ungrounded rate (95% CI) | Notes |
|---|---|---|---|---|---|
| 2026-10-15 | `configs/v0.yaml` | none (baseline) | `20261015T152638Z-v0-c7c96975` | 8.75% [4.30%, 16.98%] (7/80), uncalibrated judge | v0 baseline; run_reports id 1 |
| 2026-10-15 | `configs/v1_seed43.yaml` | generation.seed 42 -> 43 (noise control) | `20261015T215947Z-v1_seed43-cf6d47d3` | 6.25% [2.70%, 13.81%] (5/80), uncalibrated judge | -2.50 pts vs baseline; improved 2, regressed 0, p = 0.5000; false refusals 1 |
| 2026-10-15 | `configs/v1_prompt_grounded.yaml` | generation.prompt_template: answer only from the context | `20261015T220338Z-v1_prompt_grounded-3cb0651e` | 6.25% [2.70%, 13.81%] (5/80), uncalibrated judge | -2.50 pts vs baseline; improved 4, regressed 2, p = 0.6875; false refusals 1 |
| 2026-10-15 | `configs/v1_prompt_citations.yaml` | generation.prompt_template + context_chunk_template: numbered chunks and [n] citations | `20261015T220912Z-v1_prompt_citations-23b6da7c` | 2.50% [0.69%, 8.66%] (2/80), uncalibrated judge | -6.25 pts vs baseline; improved 5, regressed 0, p = 0.0625; false refusals 1 |
| 2026-10-15 | `configs/v1_prompt_refusal.yaml` | generation.prompt_template: explicit refusal sentence | `20261015T221540Z-v1_prompt_refusal-0a78ad5c` | 3.75% [1.28%, 10.45%] (3/80), uncalibrated judge | -5.00 pts vs baseline; improved 4, regressed 0, p = 0.1250; false refusals 3 |
| 2026-10-15 | `configs/v1_chunk75.yaml` | chunking.chunk_size 150 -> 75 | `20261015T221834Z-v1_chunk75-68312bcf` | 2.50% [0.69%, 8.66%] (2/80), uncalibrated judge | -6.25 pts vs baseline; improved 5, regressed 0, p = 0.0625; false refusals 2 |
| 2026-10-15 | `configs/v1_chunk300.yaml` | chunking.chunk_size 150 -> 300 | `20261015T222356Z-v1_chunk300-75ec7722` | 6.25% [2.70%, 13.81%] (5/80), uncalibrated judge | -2.50 pts vs baseline; improved 4, regressed 2, p = 0.6875; false refusals 4 |
| 2026-10-15 | `configs/v1_overlap0.yaml` | chunking.chunk_overlap 30 -> 0 | `20261015T222943Z-v1_overlap0-324866ad` | 11.25% [6.03%, 20.02%] (9/80), uncalibrated judge | +2.50 pts vs baseline; improved 4, regressed 6, p = 0.7539; false refusals 4 |
| 2026-10-15 | `configs/v1_topk2.yaml` | retrieval.top_k 4 -> 2 | `20261015T223401Z-v1_topk2-3c74a564` | 5.00% [1.96%, 12.16%] (4/80), uncalibrated judge | -3.75 pts vs baseline; improved 3, regressed 0, p = 0.2500; false refusals 4 |
| 2026-10-15 | `configs/v1_topk6.yaml` | retrieval.top_k 4 -> 6 | `20261015T223749Z-v1_topk6-217f7b9a` | 7.50% [3.48%, 15.41%] (6/80), uncalibrated judge | -1.25 pts vs baseline; improved 4, regressed 3, p = 1.0000; false refusals 1 |
| 2026-10-15 | `configs/v1_hybrid.yaml` | retrieval.method vector -> hybrid (BM25 + vector, RRF) | `20261015T224133Z-v1_hybrid-12a7c4ec` | 5.00% [1.96%, 12.16%] (4/80), uncalibrated judge | -3.75 pts vs baseline; improved 4, regressed 1, p = 0.3750; false refusals 1 |
| 2026-10-15 | `configs/v1_rerank.yaml` | retrieval.rerank.enabled false -> true (top 10 -> top 4) | `20261015T224509Z-v1_rerank-d2d11546` | 3.75% [1.28%, 10.45%] (3/80), uncalibrated judge | -5.00 pts vs baseline; improved 5, regressed 1, p = 0.2188; false refusals 2 |
| 2026-10-15 | `configs/v1_rewrite.yaml` | query_rewrite.enabled false -> true | `20261015T224841Z-v1_rewrite-a4918ecb` | 1.25% [0.22%, 6.75%] (1/80), uncalibrated judge | -7.50 pts vs baseline; improved 6, regressed 0, p = 0.0312; false refusals 4 |
| 2026-10-15 | `configs/v0.yaml` | none (held-out baseline) | `20261015T233336Z-v0-9ff3bd9f` | 10.00% [2.79%, 30.10%] (2/20), uncalibrated judge | held-out, 20 unseen queries |
| 2026-10-15 | `configs/v1_prompt_citations.yaml` | held-out confirmation | `20261015T233407Z-v1_prompt_citations-84777159` | 5.00% [0.89%, 23.61%] (1/20), uncalibrated judge | held-out verdict not_distinguishable; improved 1, regressed 0, p = 1.0000 |
