# -*- coding: utf-8 -*-
"""Calibrate the Task 9 Dense fallback gate (SCORE_THRESHOLD).

Method (repo-mandated): separate in-domain vs out-of-domain queries using
ONLY the top-1 original Dense cosine score (1 - cosine distance).
No BM25 / RRF / PageIndex / LLM signal is used for calibration.

- In-domain: all 16 golden-dataset questions (supported by the corpus).
- Out-of-domain: explicit realistic questions unanswerable from the
  9-document corpus (weapon meta, map tactics, settings, other games,
  weather, cooking, finance, general knowledge, ...).

Run: ``python group_project/evaluation/calibrate_threshold.py``
Writes: ``group_project/evaluation/threshold_calibration.json``
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.task5_semantic_search import semantic_search  # noqa: E402

EVAL_DIR = REPO / "group_project" / "evaluation"
ARTIFACT = EVAL_DIR / "threshold_calibration.json"
TOP_K_PROBE = 3

# Realistic user questions NOT answerable from the 9-document corpus.
OUT_OF_DOMAIN = [
    "Súng nào mạnh nhất trong PUBG hiện tại?",
    "M416 nên dùng attachment nào?",
    "PUBG map nào dễ thắng nhất?",
    "Cách drop dù xuống Sosnovka an toàn nhất?",
    "PUBG Mobile và PUBG PC khác nhau thế nào?",
    "Cấu hình PC bao nhiêu FPS để chơi PUBG mượt?",
    "Cách chỉnh độ nhạy chuột cho PUBG?",
    "Đội tuyển nào mạnh nhất PUBG Việt Nam hiện tại?",
    "So sánh PUBG và Free Fire game nào hay hơn?",
    "Patch notes PUBG update mới nhất có gì?",
    "Ai vô địch PGC năm nay?",
    "Valorant rank distribution hiện tại là gì?",
    "Thời tiết Hà Nội hôm nay thế nào?",
    "Cách nấu phở bò ngon nhất?",
    "Giá Bitcoin hiện tại bao nhiêu?",
    "What time is it in Hanoi right now?",
]


def load_in_domain() -> list[dict]:
    golden = json.loads((EVAL_DIR / "golden_dataset.json")
                        .read_text(encoding="utf-8"))
    return [{"id": c["id"], "question": c["question"]} for c in golden]


def collect(query: str, label: str, qid: str) -> dict:
    """Real Dense retrieval only; record top-1 (+top-3) cosine scores."""
    results = semantic_search(query, top_k=TOP_K_PROBE)
    top = results[0] if results else None
    return {
        "id": qid,
        "question": query,
        "label": label,
        "top1_score": round(top["score"], 6) if top else None,
        "top1_id": top["id"] if top else None,
        "top1_source": (top["metadata"] or {}).get("source") if top else None,
        "top3_scores": [round(r["score"], 6) for r in results],
    }


def describe(scores: list[float]) -> dict:
    scores = sorted(scores)
    n = len(scores)
    def pct(p: float) -> float:
        if n == 1:
            return scores[0]
        rank = (n - 1) * p / 100
        low, frac = int(rank), rank - int(rank)
        return scores[low] + frac * (scores[min(low + 1, n - 1)] - scores[low])
    return {
        "n": n,
        "min": round(scores[0], 4),
        "max": round(scores[-1], 4),
        "mean": round(statistics.fmean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "p10": round(pct(10), 4),
        "p25": round(pct(25), 4),
        "p75": round(pct(75), 4),
        "p90": round(pct(90), 4),
    }


def confusion(rows: list[dict], threshold: float) -> dict:
    tp = sum(1 for r in rows if r["label"] == "in_domain"
             and r["top1_score"] is not None and r["top1_score"] >= threshold)
    fn = sum(1 for r in rows if r["label"] == "in_domain"
             and not (r["top1_score"] is not None
                      and r["top1_score"] >= threshold))
    tn = sum(1 for r in rows if r["label"] == "out_of_domain"
             and not (r["top1_score"] is not None
                      and r["top1_score"] >= threshold))
    fp = sum(1 for r in rows if r["label"] == "out_of_domain"
             and r["top1_score"] is not None and r["top1_score"] >= threshold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    balanced = (recall + specificity) / 2
    return {
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round(balanced, 4),
    }


def select_threshold(rows: list[dict]) -> tuple[float, dict]:
    """Max balanced accuracy; ties -> widest-margin midpoint in the gap.

    Rule rationale: never reject valid in-domain queries excessively
    (recall), while reliably rejecting unsupported queries (specificity);
    balanced accuracy weighs both classes equally despite any size skew.
    Among all cutoffs tied at the optimum, the midpoint of the widest
    optimal gap maximizes the margin to both classes, so future queries
    near the boundary are classified more robustly than at an edge.
    """
    scores = sorted({r["top1_score"] for r in rows
                     if r["top1_score"] is not None})
    candidates = sorted({round(s, 4) for s in scores}
                        | {round(s + 0.0001, 4) for s in scores})
    scored = [(t, confusion(rows, t)) for t in candidates]
    best_key = max(
        (m["balanced_accuracy"], m["f1"], m["recall"]) for _, m in scored)
    tied = [t for t, m in scored
            if (m["balanced_accuracy"], m["f1"], m["recall"]) == best_key]
    # Widest-margin point: midpoint of the tied-optimal span. Candidate
    # cutoffs are sparse samples of continuous optimal intervals, so the
    # span endpoints (not grid contiguity) define the margin.
    midpoint = round((min(tied) + max(tied)) / 2, 4)
    matrix = confusion(rows, midpoint)
    if ((matrix["balanced_accuracy"], matrix["f1"], matrix["recall"])
            != best_key):
        midpoint = min(tied)  # rounding safety: fall back to gap edge
        matrix = confusion(rows, midpoint)
    return midpoint, matrix


def main() -> dict:
    in_domain = load_in_domain()
    rows = ([collect(c["question"], "in_domain", c["id"]) for c in in_domain]
            + [collect(q, "out_of_domain", f"ood{i:02d}")
               for i, q in enumerate(OUT_OF_DOMAIN, 1)])
    scored_in = [r["top1_score"] for r in rows
                 if r["label"] == "in_domain" and r["top1_score"] is not None]
    scored_out = [r["top1_score"] for r in rows
                  if r["label"] == "out_of_domain"
                  and r["top1_score"] is not None]
    threshold, matrix = select_threshold(rows)
    artifact = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": "text-embedding-3-small",
        "score_semantics": "top-1 original Dense cosine score "
                           "(1 - cosine distance, higher = more relevant)",
        "num_in_domain": len(in_domain),
        "num_out_of_domain": len(OUT_OF_DOMAIN),
        "selected_threshold": threshold,
        "selection_method": ("max balanced accuracy over observed score "
                             "cutoffs; ties broken by F1, then recall, then "
                             "widest-margin midpoint of the optimal gap"),
        "in_domain": describe(scored_in),
        "out_of_domain": describe(scored_out),
        "confusion_matrix": {k: matrix[k] for k in ("tp", "fn", "tn", "fp")},
        "metrics": {k: matrix[k] for k in ("precision", "recall",
                                          "specificity", "f1",
                                          "balanced_accuracy")},
        "queries": rows,
    }
    ARTIFACT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"in-domain ({len(scored_in)}):", artifact["in_domain"])
    print(f"out-of-domain ({len(scored_out)}):", artifact["out_of_domain"])
    print("selected_threshold:", threshold, matrix)
    print("saved:", ARTIFACT)
    return artifact


if __name__ == "__main__":
    main()
