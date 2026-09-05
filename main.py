"""Load lead data from a CSV file."""

import argparse
import logging
from pathlib import Path
import re
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from config import load_config
from llm import analyze_leads
from report import generate_report, write_report
from scoring import score_lead


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


EXPECTED_COLUMNS = {
    "name",
    "company",
    "company_size",
    "industry",
    "source",
    "last_interaction_date",
}


def load_leads(csv_path: Path) -> pd.DataFrame:
    """Load a lead CSV and verify its columns.

    Args:
        csv_path: Path to the lead data.

    Returns:
        A DataFrame containing the loaded leads.

    Raises:
        ValueError: If the file cannot be read or required columns are missing.
    """
    logger.info("Loading leads from %s", csv_path)
    try:
        leads = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise ValueError(f"CSV file not found: {csv_path}") from exc
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read CSV file: {csv_path}") from exc

    missing_columns = EXPECTED_COLUMNS - set(leads.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    return leads


def is_missing(value: Any) -> bool:
    """Check whether a value is empty or missing.

    Args:
        value: Value to inspect.

    Returns:
        True when the value is missing; otherwise, False.
    """
    return pd.isna(value) or (isinstance(value, str) and not value.strip())


def validate_lead(lead: pd.Series) -> dict[str, Any]:
    """Validate one lead.

    Args:
        lead: Lead record to validate.

    Returns:
        Validation status and identified issues.
    """
    issues: list[str] = []

    if is_missing(lead["name"]):
        issues.append("Missing name")

    if is_missing(lead["company"]):
        issues.append("Missing company")

    company_size = lead["company_size"]
    if is_missing(company_size):
        issues.append("Missing company size")
    else:
        numeric_size = pd.to_numeric(company_size, errors="coerce")
        if pd.isna(numeric_size) or numeric_size <= 0:
            issues.append("Invalid company size")

    if is_missing(lead["source"]):
        issues.append("Missing lead source")

    interaction_date = lead["last_interaction_date"]
    if is_missing(interaction_date):
        issues.append("Missing interaction date")
    elif not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(interaction_date)):
        issues.append("Invalid interaction date")
    else:
        try:
            pd.to_datetime(interaction_date, format="%Y-%m-%d", errors="raise")
        except (TypeError, ValueError):
            issues.append("Invalid interaction date")

    return {
        "validation_status": "review" if issues else "valid",
        "validation_issues": issues,
    }


def validate_leads(leads: pd.DataFrame) -> pd.DataFrame:
    """Validate all leads without stopping for invalid rows.

    Args:
        leads: Lead records to validate.

    Returns:
        Lead records with validation status and issues.
    """
    logger.info("Validating %d leads", len(leads))
    validation = leads.apply(validate_lead, axis=1, result_type="expand")
    return pd.concat([leads, validation], axis=1)


def qualify_lead(lead: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    """Score a lead while preserving validation review decisions.

    Args:
        lead: Validated lead record.
        config: Validated project configuration.

    Returns:
        Qualification scores, decision, priority, and explanation.
    """
    result = score_lead(lead, config)
    if lead["validation_status"] == "review":
        issues = "; ".join(lead["validation_issues"])
        result["decision"] = "review"
        result["priority"] = None
        result["explanation"] += f" Validation review required: {issues}."
    return result


def build_llm_payload(scored_leads: pd.DataFrame) -> list[dict[str, Any]]:
    """Prepare JSON-safe lead records for Ollama.

    Args:
        scored_leads: Validated and deterministically scored leads.

    Returns:
        Lead records ready for batched LLM analysis.
    """
    payload = []
    for lead_id, (_, lead) in enumerate(scored_leads.iterrows(), start=1):
        company_size = lead["company_size"]
        if not is_missing(company_size) and hasattr(company_size, "item"):
            company_size = company_size.item()

        payload.append(
            {
                "lead_id": lead_id,
                "name": None if is_missing(lead["name"]) else str(lead["name"]),
                "company": (
                    None if is_missing(lead["company"]) else str(lead["company"])
                ),
                "company_size": None if is_missing(company_size) else company_size,
                "industry": (
                    None if is_missing(lead["industry"]) else str(lead["industry"])
                ),
                "source": (
                    None if is_missing(lead["source"]) else str(lead["source"])
                ),
                "last_interaction_date": (
                    None
                    if is_missing(lead["last_interaction_date"])
                    else str(lead["last_interaction_date"])
                ),
                "validation_status": lead["validation_status"],
                "validation_issues": lead["validation_issues"],
                "deterministic_scores": {
                    **lead["factor_scores"],
                    "total": lead["total_score"],
                },
                "deterministic_decision": lead["decision"],
                "deterministic_priority": lead["priority"],
            }
        )
    return payload


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load and qualify sales leads.")
    parser.add_argument("csv_path", type=Path, help="Path to the leads CSV file")
    return parser.parse_args()


def main() -> None:
    """Run the lead validation, scoring, and LLM review workflow."""
    load_dotenv()
    args = parse_args()

    try:
        config = load_config()
        leads = load_leads(args.csv_path)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    logger.info("Loaded %d leads", len(leads))

    validated_leads = validate_leads(leads)
    status_counts = validated_leads["validation_status"].value_counts()
    valid_count = int(status_counts.get("valid", 0))
    review_count = int(status_counts.get("review", 0))
    logger.info(
        "Validation complete: %d valid, %d review", valid_count, review_count
    )

    review_leads = validated_leads[validated_leads["validation_status"] == "review"]
    for index, lead in review_leads.iterrows():
        name = "<missing>" if is_missing(lead["name"]) else lead["name"]
        company = "<missing>" if is_missing(lead["company"]) else lead["company"]
        issues = "; ".join(lead["validation_issues"])
        logger.warning(
            "Review required | CSV row %d | name=%s | company=%s | issues=%s",
            index + 2,
            name,
            company,
            issues,
        )

    scoring = validated_leads.apply(
        lambda lead: qualify_lead(lead, config), axis=1, result_type="expand"
    )
    scored_leads = pd.concat([validated_leads, scoring], axis=1)
    decision_counts = scored_leads["decision"].value_counts()
    logger.info(
        "Scoring complete: %d qualified, %d review, %d rejected",
        int(decision_counts.get("qualified", 0)),
        int(decision_counts.get("review", 0)),
        int(decision_counts.get("rejected", 0)),
    )

    try:
        llm_results = analyze_leads(build_llm_payload(scored_leads), config)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"LLM analysis failed: {exc}") from exc

    llm_frame = pd.DataFrame(llm_results)
    final_leads = pd.concat(
        [scored_leads.reset_index(drop=True), llm_frame], axis=1
    )
    agreements = int(final_leads["agrees_with_baseline"].sum())
    agreement_rate = agreements / len(final_leads) * 100
    logger.info(
        "LLM comparison complete: %d/%d agree (%.1f%%)",
        agreements,
        len(final_leads),
        agreement_rate,
    )
    logger.info(
        "LLM safeguards: %d reasoning fallbacks, %d failed-batch reviews, "
        "%d validation skips",
        int((final_leads["reasoning_source"] == "deterministic_fallback").sum()),
        int((final_leads["decision_source"] == "batch_fallback").sum()),
        int((final_leads["decision_source"] == "validation").sum()),
    )
    logger.info(
        "Generated %d grounded outreach messages",
        int((final_leads["outreach_source"] == "grounded_template").sum()),
    )

    output_path = Path("output_report.json")
    write_report(generate_report(final_leads), output_path)
    logger.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
