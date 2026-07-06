"""
eval_run.py — runs the gold set through the pipeline, checks grounding,
and prints a faithfulness report. Save results as timestamped JSON so you
can show grounding scores improving over time in interviews.

Metrics produced:
  - citation_hit_rate     : % of expected citations that appeared in generated claims
  - avg_grounding_conf    : mean grounding_confidence across all claims in the run
  - uncited_claim_rate    : % of generated claims flagged "uncited" (hallucination risk)
  - low_conf_claim_rate   : % of claims below CONFIDENCE_THRESHOLD
  - out_of_scope_accuracy : did out-of-scope docs get correctly refused?
"""

import json
import os
import sys
import re
from datetime import datetime
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from pipeline import LegalLensPipeline

GOLD_SETS = [
    os.path.join(os.path.dirname(__file__), "eval", "cheque_bounce_gold.json"),
    os.path.join(os.path.dirname(__file__), "eval", "eviction_notice_gold.json"),
    os.path.join(os.path.dirname(__file__), "eval", "fir_gold.json"),
    os.path.join(os.path.dirname(__file__), "eval", "rental_agreement_gold.json"),
    os.path.join(os.path.dirname(__file__), "eval", "consumer_complaint_gold.json")
]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "eval", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def normalize_section(citation: str) -> str:
    # extract just the leading section number e.g. "27" from "27(2)(a)" or "138" from "143A"
    m = re.search(r'\b(\d+[A-Z]?)', citation)
    return m.group(1).upper() if m else citation.strip().upper()

def run_eval():
    gold_set = []
    for path in GOLD_SETS:
        with open(path) as f:
            gold_set.extend(json.load(f))

    pipeline = LegalLensPipeline()

    all_results = []
    citation_hits, citation_total = 0, 0
    all_grounding_confs = []
    uncited_count, total_claims = 0, 0
    low_conf_count = 0
    oos_correct, oos_total = 0, 0

    print("\n=== LegalLens Eval Run ===\n")

    for entry in gold_set:
        doc_id = entry["id"]
        is_oos = entry.get("out_of_scope_check", False)
        print(f"[{doc_id}] {'OUT-OF-SCOPE' if is_oos else 'in-scope'}", end=" ... ")

        result = pipeline.run(entry["document_text"])
        time.sleep(15)  # avoid Groq TPM rate limit

        entry_result = {
            "id": doc_id,
            "out_of_scope_check": is_oos,
            "pipeline_out_of_scope": result["out_of_scope"],
            "claims": result["claims"],
            "avg_grounding_confidence": result["avg_grounding_confidence"],
            "low_confidence_warning": result["low_confidence_warning"],
        }

        if is_oos:
            oos_total += 1
            if result["out_of_scope"]:
                oos_correct += 1
                print("✅ correctly refused")
            else:
                print("❌ should have been refused but got claims")
            all_results.append(entry_result)
            continue

        generated_sections = set(
            normalize_section(c["full_citation"])
            for c in result["claims"]
        )
        expected_sections = set(
            normalize_section(ec.get("expected_citation", ec.get("expected_section", "")))
            for ec in entry.get("expected_claims", [])
        )

        hits = len(expected_sections & generated_sections)
        citation_hits += hits
        citation_total += len(expected_sections)

        for claim in result["claims"]:
            total_claims += 1
            conf = claim["grounding_confidence"]
            all_grounding_confs.append(conf)
            if claim["confidence_flag"] == "uncited":
                uncited_count += 1
            elif claim["confidence_flag"] == "low":
                low_conf_count += 1

        avg_conf = result["avg_grounding_confidence"]
        print(f"citations {hits}/{len(expected_sections)} | avg_conf={avg_conf:.2f} | {'⚠ LOW CONF' if result['low_confidence_warning'] else '✅'}")
        all_results.append(entry_result)

    citation_hit_rate = citation_hits / citation_total if citation_total else 0
    avg_grounding_conf = sum(all_grounding_confs) / len(all_grounding_confs) if all_grounding_confs else 0
    uncited_rate = uncited_count / total_claims if total_claims else 0
    low_conf_rate = low_conf_count / total_claims if total_claims else 0
    oos_accuracy = oos_correct / oos_total if oos_total else 0

    summary = {
        "run_timestamp": datetime.now().isoformat() + "Z",
        "citation_hit_rate": round(citation_hit_rate, 3),
        "avg_grounding_confidence": round(avg_grounding_conf, 3),
        "uncited_claim_rate": round(uncited_rate, 3),
        "low_conf_claim_rate": round(low_conf_rate, 3),
        "out_of_scope_accuracy": round(oos_accuracy, 3),
        "total_documents": len(gold_set),
        "total_claims_generated": total_claims,
    }

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        if k != "run_timestamp":
            print(f"  {k:<30} {v}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"eval_run_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_document": all_results}, f, indent=2)
    print(f"\nResults saved → {out_path}")
    return summary


if __name__ == "__main__":
    run_eval()