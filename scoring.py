"""Deterministic scoring functions for lead qualification."""

from datetime import date
from typing import Any, Mapping


def score_company_size(company_size: Any, config: dict[str, Any]) -> int:
    """Score a lead's company size.

    Args:
        company_size: Number of employees reported for the lead.
        config: Validated project configuration.

    Returns:
        The configured company-size score, or zero for an invalid value.
    """
    try:
        size = float(company_size)
    except (TypeError, ValueError):
        return 0

    for band in config["qualification"]["company_size"]["bands"]:
        maximum = band["max"]
        if size >= band["min"] and (maximum is None or size <= maximum):
            return int(band["score"])
    return 0


def score_industry(industry: Any, config: dict[str, Any]) -> int:
    """Score a lead's industry fit.

    Args:
        industry: Industry reported for the lead.
        config: Validated project configuration.

    Returns:
        The configured industry-fit score.
    """
    rules = config["qualification"]["industry_fit"]
    normalized_industry = str(industry).strip().casefold()

    for group in ("core", "adjacent"):
        industries = {value.casefold() for value in rules[group]}
        if normalized_industry in industries:
            return int(rules["scores"][group])
    return int(rules["scores"]["default"])


def score_lead_source(source: Any, config: dict[str, Any]) -> int:
    """Score how a lead was acquired.

    Args:
        source: Acquisition source reported for the lead.
        config: Validated project configuration.

    Returns:
        The configured lead-source score.
    """
    rules = config["qualification"]["lead_source"]
    source_scores = {
        name.casefold(): score for name, score in rules["scores"].items()
    }
    normalized_source = str(source).strip().casefold()
    return int(source_scores.get(normalized_source, rules["default_score"]))


def score_recency(interaction_date: Any, config: dict[str, Any]) -> int:
    """Score the recency of a lead's last interaction.

    Args:
        interaction_date: Last interaction date in ISO format.
        config: Validated project configuration.

    Returns:
        The configured recency score, or zero for an invalid date.
    """
    try:
        interaction = date.fromisoformat(str(interaction_date))
        evaluation = date.fromisoformat(config["evaluation_date"])
    except (TypeError, ValueError):
        return 0

    days_since_interaction = (evaluation - interaction).days
    for band in config["qualification"]["interaction_recency"]["bands"]:
        maximum = band["max_days"]
        if days_since_interaction >= band["min_days"] and (
            maximum is None or days_since_interaction <= maximum
        ):
            return int(band["score"])
    return 0


def score_lead(lead: Mapping[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Calculate a lead's factor scores and qualification outcome.

    Args:
        lead: Lead record containing the four qualification inputs.
        config: Validated project configuration.

    Returns:
        Factor scores, total score, decision, priority, and explanation.

    Raises:
        ValueError: If no configured threshold matches the total score.
    """
    factor_scores = {
        "company_size": score_company_size(lead["company_size"], config),
        "industry_fit": score_industry(lead["industry"], config),
        "lead_source": score_lead_source(lead["source"], config),
        "interaction_recency": score_recency(
            lead["last_interaction_date"], config
        ),
    }
    total_score = sum(factor_scores.values())

    outcome = next(
        (
            threshold
            for threshold in config["thresholds"]
            if threshold["min_score"] <= total_score <= threshold["max_score"]
        ),
        None,
    )
    if outcome is None:
        raise ValueError(f"No threshold configured for score {total_score}")

    explanation = (
        f"Company size {factor_scores['company_size']}/3, "
        f"industry fit {factor_scores['industry_fit']}/2, "
        f"lead source {factor_scores['lead_source']}/3, "
        f"recency {factor_scores['interaction_recency']}/2."
    )
    return {
        "factor_scores": factor_scores,
        "total_score": total_score,
        "decision": outcome["decision"],
        "priority": outcome["priority"],
        "explanation": explanation,
    }
