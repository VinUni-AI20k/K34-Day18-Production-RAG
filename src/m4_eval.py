from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}
             
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        import pandas as pd
        import numpy as np
        
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
                                            
        df = result.to_pandas()
        
        per_question = []
        for _, row in df.iterrows():
            f = float(row.get("faithfulness", 0.0))
            ar = float(row.get("answer_relevancy", 0.0))
            cp = float(row.get("context_precision", 0.0))
            cr = float(row.get("context_recall", 0.0))
            
            per_question.append(EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=row["contexts"],
                ground_truth=row["ground_truth"],
                faithfulness=f if not np.isnan(f) else 0.0,
                answer_relevancy=ar if not np.isnan(ar) else 0.0,
                context_precision=cp if not np.isnan(cp) else 0.0,
                context_recall=cr if not np.isnan(cr) else 0.0
            ))
            
        f_agg = result.get("faithfulness", 0.0)
        ar_agg = result.get("answer_relevancy", 0.0)
        cp_agg = result.get("context_precision", 0.0)
        cr_agg = result.get("context_recall", 0.0)
        
        return {
            "faithfulness": float(f_agg) if not np.isnan(f_agg) else 0.0,
            "answer_relevancy": float(ar_agg) if not np.isnan(ar_agg) else 0.0,
            "context_precision": float(cp_agg) if not np.isnan(cp_agg) else 0.0,
            "context_recall": float(cr_agg) if not np.isnan(cr_agg) else 0.0,
            "per_question": per_question
        }
        
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    
    if not eval_results:
        return []
        
    analyzed = []
    for res in eval_results:
        metrics = {
            "faithfulness": res.faithfulness,
            "answer_relevancy": res.answer_relevancy,
            "context_precision": res.context_precision,
            "context_recall": res.context_recall
        }
        
        avg_score = sum(metrics.values()) / 4.0
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]
        
        diag, fix = diagnostic_tree[worst_metric]
        
        analyzed.append({
            "question": res.question,
            "worst_metric": worst_metric,
            "score": worst_score,
            "avg_score": avg_score,
            "diagnosis": diag,
            "suggested_fix": fix
        })
        
    analyzed.sort(key=lambda x: x["avg_score"])
    
    for a in analyzed:
        a.pop("avg_score", None)
        
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
