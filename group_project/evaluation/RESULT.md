# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 (evaluated_at 2026-09-25T17:23:50+00:00, q12B backfilled after one transient retry) |
| Framework and version              | Repo script `group_project/evaluation/evaluate.py`; semantic LLM-judge metrics with disk cache (`.judge_cache.json`, gitignored) |
| Evaluator model                    | gpt-4.1-mini, temperature 0, JSON-object mode, max 800 tokens; same judge model, prompt style and parameters for Config A and Config B |
| Generator model                    | gpt-4.1-mini (Task 10 default; shared by both configs) |
| Embedding model                    | text-embedding-3-small (dim 1536, Chroma collection `rag_documents`) |
| Corpus version/commit              | 4ebc104, 274 chunks (9 standardized docs: 5 vi + 4 en) |
| Golden dataset size                | 16 (`group_project/evaluation/golden_dataset.json`) |
| `top_k`                            | 5 (both configs; RRF k=60) |
| Fallback threshold and calibration | 0.3 (repo default, not calibrated in this run) |

Metric definitions (all judged claim-by-claim against retrieved contexts; cross-lingual paraphrase counts as equivalent):
faithfulness = supported answer claims / total answer claims (safe refusal = 1.0, it makes no factual claims);
answer relevance = judge score 0–1 for directness and completeness;
context recall = supported reference claims (from expected answer + expected context) / total reference claims;
context precision = retrieved chunks materially helping answer the question / retrieved chunks (rank order preserved in detailed rows).
Each row stores judge verdicts with concise reason/evidence only (no chain-of-thought) plus token-overlap diagnostics.

## Configurations

- **Config A — dense-only:** `task5.semantic_search → top_k=5`
- **Config B — hybrid + RRF:** `task5.semantic_search` + `task6.lexical_search` (depth 10) → `task7.rerank_rrf(top_k=5, k=60)`

Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |   0.9792 |   1.0000 |   +0.0208 |
| Answer relevance  |   0.6750 |   0.9000 |   +0.2250 |
| Context recall    |   0.9531 |   0.9688 |   +0.0157 |
| Context precision |   0.4375 |   0.5500 |   +0.1125 |
| **Average**       |   0.7612 |   0.8797 |   +0.1185 |

Averages over 16 golden cases × config (32 rows in `evaluation_results.json`).
Refusals: Config A refused 5/16 (dense-only context too weak → safe refusal);
Config B refused 1/16. One transient judge ConnectionError occurred during the run and was retried cleanly; final rows are 100% judge-scored with zero fallbacks.

## A/B comparison

- Cấu hình tốt hơn: Config B (hybrid + RRF), average +0.1185.
- Evidence: relevance +0.2250 and precision +0.1125 (BM25 surfaces exact-match chunks dense misses: q01/q02/q06/q08 answered under B while refused under A); recall +0.0157 with both configs near ceiling; faithfulness at ceiling for B (1.0000). Cross-lingual pairs now score correctly: q13–q16 faithfulness rose from 0.02–0.19 (old token-overlap) to 1.0000 under the judge.
- Trade-off về latency/cost: B costs one extra query-embedding call plus in-memory BM25 per query plus ~2× judge calls at eval time; generation cost identical (same model, same top_k). Judge calls are disk-cached, so reruns cost nothing.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | q02 teamkill penalty | A | 1.000 | 0.000 | 0.625 | 0.200 | retrieval | Dense-only context too weak (recall 0.625) → correct safe refusal; hybrid answers it (recall 1.0) |
|   2 | q01 anti-cheat rules | A | 1.000 | 0.000 | 0.875 | 0.000 | retrieval | Dense-only retrieved no materially helpful chunk (precision 0.0) → correct refusal; hybrid reaches recall 1.0 |
|   3 | q06 game license terms | A | 1.000 | 0.000 | 1.000 | 0.200 | retrieval | Evidence present but ranked uselessly for dense-only → refusal; hybrid answers with relevance 1.0 |

Notable non-table case: q05 (G-COIN refund) refused under both configs despite recall 0.875 — generation-side conservatism (citation validation), not a retrieval gap.

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Calibrate SCORE_THRESHOLD on in-domain vs out-of-domain queries | 5/16 dense-only refusals vs 1/16 hybrid at uncalibrated 0.3 | Fewer false refusals without hurting precision | Sweep threshold on golden + out-of-domain probes, record refusal/precision curve |
|        2 | Log rejected generation drafts (with reason: no-citation/out-of-range/link) for refusal cases | q05 refused with recall 0.875 and no diagnosis trail | Distinguish retrieval gaps from over-strict validation | Add draft+reason fields to evaluation rows, rerun |
|        3 | Raise useful density of Top-5 (precision 0.44–0.55: only ~half the chunks materially help) | Precision is the lowest metric in both configs | Higher precision without changing top_k | Experiment with retrieval depth and chunking overlap, re-run this same judge suite |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| (none) | — | — | — | Không thực hiện thí nghiệm bonus; ngoài phạm vi lab |

## Fallback threshold calibration

- Embedding model: text-embedding-3-small; signal: top-1 original Dense cosine score only.
- Calibration data: 16 in-domain (all golden questions) + 16 out-of-domain realistic probes (weapon meta, map tactics, settings, other games, weather, cooking, finance, general knowledge).
- Score ranges: in-domain 0.6028–0.7973 (mean 0.6909); out-of-domain 0.2510–0.5428 (mean 0.4081); zero overlap.
- Selected threshold: **0.5728** (midpoint of the gap; max balanced accuracy with widest-margin tie-break).
- Result on calibration data: TP 16 / FN 0 / TN 16 / FP 0 → recall 1.0, specificity 1.0, F1 1.0, balanced accuracy 1.0. In-domain recall 16/16; out-of-domain rejection 16/16.
- Known limitation: calibrated on 32 queries for one embedding model/index snapshot; re-calibrate if the corpus, chunking, or embedding model changes. Full per-query scores: `group_project/evaluation/threshold_calibration.json`; method: `group_project/evaluation/calibrate_threshold.py`.
