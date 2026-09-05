"""JSON report generation for processed leads."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def _clean(value: Any) -> Any:
    """Convert pandas values into JSON-safe values."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _get_rejection_reasons(factor_scores: dict[str, int]) -> list[str]:
    """Return deterministic reasons that contributed to rejection."""
    reasons = []
    if factor_scores["company_size"] == 0:
        reasons.append("Low company-size fit")
    if factor_scores["industry_fit"] == 0:
        reasons.append("Low industry fit")
    if factor_scores["lead_source"] == 0:
        reasons.append("Low-intent lead source")
    if factor_scores["interaction_recency"] == 0:
        reasons.append("Stale or unclear interaction")
    return reasons


def generate_report(leads: pd.DataFrame) -> dict[str, Any]:
    """Build the final lead qualification report.

    Args:
        leads: Fully processed lead records.

    Returns:
        Report containing summary metrics, lead results, and outreach examples.
    """
    records = []
    rejection_reasons: Counter[str] = Counter()

    for _, lead in leads.iterrows():
        # Disagreements and processing failures require human review.
        final_decision = lead["llm_decision"]
        if not lead["agrees_with_baseline"] or lead["decision_source"] in {
            "validation",
            "batch_fallback",
        }:
            final_decision = "review"

        priority = lead["baseline_priority"] if final_decision == "qualified" else None
        outreach = lead["outreach_message"] if final_decision == "qualified" else None
        factor_scores = lead["factor_scores"]

        if final_decision == "rejected":
            rejection_reasons.update(_get_rejection_reasons(factor_scores))

        records.append(
            {
                "lead_id": int(lead["lead_id"]),
                "name": _clean(lead["name"]),
                "company": _clean(lead["company"]),
                "industry": _clean(lead["industry"]),
                "source": _clean(lead["source"]),
                "score": int(lead["total_score"]),
                "factor_scores": factor_scores,
                "decision": final_decision,
                "priority": priority,
                "reasoning": lead["llm_reasoning"],
                "outreach_message": outreach,
                "validation_status": lead["validation_status"],
                "validation_issues": lead["validation_issues"],
                "deterministic_decision": lead["decision"],
                "llm_decision": lead["llm_decision"],
                "agrees_with_baseline": bool(lead["agrees_with_baseline"]),
                "reasoning_source": lead["reasoning_source"],
                "outreach_source": lead["outreach_source"],
            }
        )

    decision_counts = Counter(record["decision"] for record in records)
    total = len(records)
    qualified = decision_counts["qualified"]
    outreach_examples = [
        {
            "lead_id": record["lead_id"],
            "name": record["name"],
            "company": record["company"],
            "message": record["outreach_message"],
        }
        for record in records
        if record["outreach_message"]
    ][:5]

    return {
        "summary": {
            "total_processed": total,
            "qualified": qualified,
            "review": decision_counts["review"],
            "rejected": decision_counts["rejected"],
            "qualified_percentage": round(qualified / total * 100, 1) if total else 0,
            "llm_agreement_percentage": round(
                sum(record["agrees_with_baseline"] for record in records)
                / total
                * 100,
                1,
            )
            if total
            else 0,
            "common_rejection_reasons": [
                {"reason": reason, "count": count}
                for reason, count in rejection_reasons.most_common()
            ],
        },
        "leads": records,
        "sample_outreach_messages": outreach_examples,
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    """Write a report to a JSON file.

    Args:
        report: Report data to write.
        output_path: Destination JSON path.
    """
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
