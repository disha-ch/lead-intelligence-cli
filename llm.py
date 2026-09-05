"""Ollama integration for structured lead review."""

import json
import logging
import os
import re
from datetime import date
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a sales lead qualification assistant.

Review leads using the provided deterministic results and business rubric.

Rules:
- Do not ignore or arbitrarily recalculate the deterministic score.
- Ground every conclusion in the supplied lead details and factor scores.
- Do not invent products, capabilities, intent, dates, or missing information.
- Keep leads with invalid or missing information under review.
- Keep reasoning qualitative, under 25 words, and grounded in supplied signals.
- Do not mention numbers, points, score bands, totals, or thresholds in reasoning.
- Return only JSON matching the supplied schema.
"""

NUMERIC_REASONING_TERMS = ("score", "point", "band", "total", "threshold")

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "lead_id": {"type": ["integer", "string"]},
                    "llm_decision": {
                        "type": "string",
                        "enum": ["qualified", "rejected", "review"],
                    },
                    "llm_reasoning": {
                        "type": "string",
                        "maxLength": 180,
                    },
                },
                "required": [
                    "lead_id",
                    "llm_decision",
                    "llm_reasoning",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def build_grounded_outreach(
    lead: dict[str, Any], config: dict[str, Any]
) -> str:
    """Build one of the configured grounded outreach variants.

    Args:
        lead: Original lead submitted to Ollama.
        config: Validated project configuration.

    Returns:
        A neutral message containing only supplied lead information.
    """
    first_name = str(lead["name"]).split()[0]
    company = str(lead["company"])
    industry = str(lead["industry"])
    source = str(lead["source"]).casefold()
    outreach_config = config["outreach"]
    activity = outreach_config["source_activities"].get(
        source, "recent interaction"
    )
    templates = outreach_config["templates"]
    try:
        variant_index = (int(lead["lead_id"]) - 1) % len(templates)
    except (TypeError, ValueError):
        variant_index = sum(map(ord, str(lead["lead_id"]))) % len(templates)
    return templates[variant_index].format(
        first_name=first_name,
        company=company,
        industry=industry,
        activity=activity,
    )


def _build_grounded_reasoning(lead: dict[str, Any]) -> str:
    """Build qualitative reasoning from deterministic factor scores.

    Args:
        lead: Lead containing deterministic scores and a baseline decision.

    Returns:
        A concise explanation grounded in deterministic signals.
    """
    scores = lead["deterministic_scores"]
    signals = [
        "Strong company-size fit" if scores["company_size"] >= 2 else "Limited company-size fit",
        "strong industry alignment" if scores["industry_fit"] == 2 else "limited industry alignment",
        "high-intent lead source" if scores["lead_source"] >= 2 else "lower-intent lead source",
        "recent engagement" if scores["interaction_recency"] == 2 else "older or unclear engagement",
    ]
    return f"{', '.join(signals)} support the {lead['deterministic_decision']} baseline."


def _validate_reasoning(review: dict[str, Any]) -> None:
    """Validate that LLM reasoning is concise and qualitative.

    Args:
        review: Structured result returned by Ollama.

    Raises:
        ValueError: If reasoning is missing, long, or contains numeric scoring claims.
    """
    reasoning = review.get("llm_reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("LLM reasoning is missing")
    if len(reasoning.split()) > 25:
        raise ValueError("LLM reasoning exceeds 25 words")

    normalized_reasoning = reasoning.casefold()
    if re.search(r"\d", reasoning) or any(
        re.search(rf"\b{term}s?\b", normalized_reasoning)
        for term in NUMERIC_REASONING_TERMS
    ):
        raise ValueError("LLM reasoning contains numeric scoring claims")


def _analyze_batch(
    leads: list[dict[str, Any]], config: dict[str, Any], attempt: int = 0
) -> list[dict[str, Any]]:
    """Review one batch of leads with the configured Ollama model.

    Args:
        leads: Validated leads with deterministic scoring results.
        config: Validated project configuration.
        attempt: Zero-based request attempt used to vary generation.

    Returns:
        Structured LLM review results for the supplied leads.

    Raises:
        ValueError: If the batch size or model response is invalid.
        RuntimeError: If Ollama cannot complete the request.
    """
    if not leads or len(leads) > config["batch_size"]:
        raise ValueError(f"Batch must contain 1-{config['batch_size']} leads")

    submitted_leads = {str(lead["lead_id"]): lead for lead in leads}
    if len(submitted_leads) != len(leads):
        raise ValueError("Each lead_id in a batch must be unique")

    evaluation_date = date.fromisoformat(config["evaluation_date"])
    grounded_leads = []
    for lead in leads:
        grounded_lead = dict(lead)
        try:
            interaction_date = date.fromisoformat(str(lead["last_interaction_date"]))
            grounded_lead["days_since_interaction"] = (
                evaluation_date - interaction_date
            ).days
        except (KeyError, TypeError, ValueError):
            grounded_lead["days_since_interaction"] = None
        grounded_leads.append(grounded_lead)

    model = os.getenv("OLLAMA_MODEL", "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    prompt = {
        "task": "Review each baseline decision and give brief qualitative reasoning.",
        "evaluation_date": config["evaluation_date"],
        "rubric": {
            "factor_maximums": {
                "company_size": 3,
                "industry_fit": 2,
                "lead_source": 3,
                "interaction_recency": 2,
            },
            "thresholds": config["thresholds"],
        },
        "leads": grounded_leads,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        "format": RESPONSE_SCHEMA,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "seed": 42 + attempt,
            "num_ctx": 8192,
            "num_predict": min(1100, max(120, len(leads) * 90 + attempt * 100)),
        },
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            api_response = json.load(response)
        result = json.loads(api_response["message"]["content"])
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Ollama returned an invalid structured response") from exc

    if len(result.get("results", [])) != len(leads):
        raise ValueError("Ollama response count does not match the submitted batch")

    for review in result["results"]:
        baseline = submitted_leads.get(str(review.get("lead_id")))
        if baseline is None:
            raise ValueError("Ollama returned an unknown lead_id")

        if baseline["validation_status"] == "review":
            review["llm_decision"] = "review"

        try:
            _validate_reasoning(review)
            review["reasoning_source"] = "llm"
        except ValueError as exc:
            logger.warning("Using grounded reasoning fallback: %s", exc)
            review["llm_reasoning"] = _build_grounded_reasoning(baseline)
            review["reasoning_source"] = "deterministic_fallback"

        if review["llm_decision"] == "qualified":
            review["outreach_message"] = build_grounded_outreach(baseline, config)
            review["outreach_source"] = "grounded_template"
        else:
            review["outreach_message"] = None
            review["outreach_source"] = None

        review["baseline_decision"] = baseline["deterministic_decision"]
        review["baseline_priority"] = baseline["deterministic_priority"]
        review["agrees_with_baseline"] = (
            review["llm_decision"] == baseline["deterministic_decision"]
        )
        review["decision_source"] = "llm"

    return result["results"]


def _build_failed_batch_result(lead: dict[str, Any]) -> dict[str, Any]:
    """Create a safe review result after both batch attempts fail.

    Args:
        lead: Lead from the failed Ollama batch.

    Returns:
        A review result that preserves the deterministic baseline.
    """
    return {
        "lead_id": lead["lead_id"],
        "llm_decision": "review",
        "llm_reasoning": "Ollama analysis was unavailable, so human review is required.",
        "reasoning_source": "batch_failure",
        "agrees_with_baseline": lead["deterministic_decision"] == "review",
        "outreach_message": None,
        "outreach_source": None,
        "baseline_decision": lead["deterministic_decision"],
        "baseline_priority": lead["deterministic_priority"],
        "decision_source": "batch_fallback",
    }


def analyze_leads(
    leads: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Review valid leads in batches while preserving invalid reviews.

    Args:
        leads: Validated leads with deterministic scoring results.
        config: Validated project configuration.

    Returns:
        LLM or validation results in the original lead order.

    Raises:
        ValueError: If lead IDs are missing or duplicated.
    """
    if not leads:
        return []

    lead_ids = [str(lead["lead_id"]) for lead in leads]
    if len(set(lead_ids)) != len(lead_ids):
        raise ValueError("Each lead_id must be unique")

    results_by_id = {}
    valid_leads = []
    for lead in leads:
        if lead["validation_status"] == "review":
            results_by_id[str(lead["lead_id"])] = {
                "lead_id": lead["lead_id"],
                "llm_decision": "review",
                "llm_reasoning": _build_grounded_reasoning(lead),
                "reasoning_source": "validation",
                "agrees_with_baseline": lead["deterministic_decision"] == "review",
                "outreach_message": None,
                "outreach_source": None,
                "baseline_decision": lead["deterministic_decision"],
                "baseline_priority": lead["deterministic_priority"],
                "decision_source": "validation",
            }
        else:
            valid_leads.append(lead)

    batch_size = config["batch_size"]
    total_batches = (len(valid_leads) + batch_size - 1) // batch_size
    for start in range(0, len(valid_leads), batch_size):
        batch = valid_leads[start : start + batch_size]
        logger.info(
            "Processing Ollama batch %d/%d with %d leads",
            start // batch_size + 1,
            total_batches,
            len(batch),
        )
        try:
            batch_results = _analyze_batch(batch, config)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Ollama batch failed; retrying once: %s", exc)
            try:
                batch_results = _analyze_batch(batch, config, attempt=1)
            except (RuntimeError, ValueError) as retry_exc:
                logger.error("Ollama batch retry failed: %s", retry_exc)
                batch_results = [_build_failed_batch_result(lead) for lead in batch]

        for result in batch_results:
            results_by_id[str(result["lead_id"])] = result

    return [results_by_id[lead_id] for lead_id in lead_ids]
