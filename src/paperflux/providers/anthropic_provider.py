"""Anthropic (Claude) backend: native PDF-in-context + structured JSON output.

Claude reads PDFs natively, so there is no managed vector store / file_search
equivalent — the whole PDF is sent as a base64 ``document`` block. Page numbers
are captured as a field in the structured JSON schema (citations and structured
output are mutually exclusive in the Messages API, so we keep the strict schema
for parity with the OpenAI path). Requests stream because the default
``max_output_tokens`` (32768) exceeds the SDK's non-streaming ceiling.
"""

import base64
import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from anthropic import AsyncAnthropic

from ..config import Config
from .base import (
    ProgressCallback,
    dump_failed_response,
    load_template,
    load_text_file,
    multi_category_schema,
    normalize_category_bundle,
    resolve_config_path,
)

logger = logging.getLogger(__name__)

# JSON Schema keywords the Anthropic structured-output engine does not accept.
_UNSUPPORTED_SCHEMA_KEYS = ("maxItems", "minimum", "maximum", "minItems")


def _strip_unsupported_schema_keys(node: Any) -> None:
    """Recursively remove schema keywords unsupported by Anthropic structured outputs.

    Constraints encoded in the stripped keywords (quote limits, page ranges) are
    also expressed in the prompt text, so dropping them from the schema does not
    change enforced behavior.
    """
    if isinstance(node, dict):
        for key in _UNSUPPORTED_SCHEMA_KEYS:
            node.pop(key, None)
        for value in node.values():
            _strip_unsupported_schema_keys(value)
    elif isinstance(node, list):
        for item in node:
            _strip_unsupported_schema_keys(item)


def _anthropic_safe_schema(max_quotes_per_category: int) -> dict:
    """Return a deep copy of the multi-category schema with unsupported keywords stripped."""
    schema = copy.deepcopy(multi_category_schema(max_quotes_per_category))
    _strip_unsupported_schema_keys(schema)
    return schema


def _thinking_and_effort(reasoning_effort: str) -> Tuple[dict, Optional[str]]:
    """Map ui.reasoning_effort to Anthropic (thinking, effort).

    "none" disables thinking and sends no effort; every other level enables
    adaptive thinking and passes the level through as the effort.
    """
    if reasoning_effort == "none":
        return {"type": "disabled"}, None
    return {"type": "adaptive"}, reasoning_effort


def _ensure_message_completed(message: Any, context: str, max_output_tokens: int) -> None:
    """Raise a descriptive error if Claude truncated or refused."""
    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == "max_tokens":
        raise ValueError(
            f"{context} response was truncated (stop_reason='max_tokens'). "
            f"Consider increasing ui.max_output_tokens (currently {max_output_tokens}) "
            "or lowering ui.reasoning_effort, detail_level, category count, or "
            "rag.max_quotes_per_category."
        )
    if stop_reason == "refusal":
        raise ValueError(f"{context} response was refused by the model.")


def _extract_text(message: Any) -> str:
    """Return the concatenated text blocks (skipping thinking blocks)."""
    chunks = []
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if text:
                chunks.append(text)
    return "".join(chunks)


class AnthropicProvider:
    """Analyze a PDF with Claude via the Messages API (native PDF in context)."""

    async def analyze_pdf(
        self,
        path: Path,
        cfg: Config,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> dict:
        """Extract quotes and a summary from a PDF using the Anthropic Messages API.

        The PDF is sent as a base64-encoded document block so Claude reads it
        natively.  Two streaming requests are made: the first extracts per-category
        quotes with structured JSON output; the second synthesizes a free-text
        summary from those category-level results.

        Args:
            path: Local path to the PDF file to analyze.
            cfg: Resolved application configuration, including model selection,
                token limits, and extraction settings.
            progress_callback: Optional callable that receives a human-readable
                status string at key stages of processing.

        Returns:
            A dict with keys ``"key_takeaways"`` (str) and ``"quotes"`` (list).

        Raises:
            ValueError: If a model response is truncated, refused, missing text
                output, or contains malformed JSON.
        """
        client = AsyncAnthropic(api_key=cfg.anthropic.api_key)

        if progress_callback:
            progress_callback(f"Reading and encoding {path.name}")
        pdf_b64 = base64.standard_b64encode(path.read_bytes()).decode("utf-8")

        categories = cfg.extraction_categories.categories
        category_template = load_template(
            resolve_config_path(cfg.rag.category_prompt_file, cfg)
        )
        category_system_prompt = load_text_file(
            resolve_config_path(cfg.rag.category_system_prompt_file_anthropic, cfg)
        )
        category_entries = [
            {"name": cat, "description": desc} for cat, desc in categories.items()
        ]
        user_msg = category_template.render(
            categories=category_entries,
            max_quotes_per_category=cfg.rag.max_quotes_per_category,
        )

        thinking, effort = _thinking_and_effort(cfg.ui.reasoning_effort)

        # Stable instructions are cached and placed before the volatile PDF so
        # the system prompt + category instructions are reused across PDFs in a
        # batch run.
        system = [
            {
                "type": "text",
                "text": category_system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        category_content = [
            {
                "type": "text",
                "text": user_msg,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_b64,
                },
            },
        ]

        category_output_config: Dict[str, Any] = {
            "format": {
                "type": "json_schema",
                "schema": _anthropic_safe_schema(cfg.rag.max_quotes_per_category),
            }
        }
        if effort:
            category_output_config["effort"] = effort

        if progress_callback:
            progress_callback("Extracting quotes with Claude")
        async with client.messages.stream(
            model=cfg.anthropic.model,
            max_tokens=cfg.ui.max_output_tokens,
            system=system,
            messages=[{"role": "user", "content": category_content}],
            thinking=thinking,
            output_config=category_output_config,
        ) as stream:
            category_message = await stream.get_final_message()
        _ensure_message_completed(
            category_message, "Category bundle", cfg.ui.max_output_tokens
        )

        text_val = _extract_text(category_message)
        if not text_val or not text_val.strip():
            dump_path = dump_failed_response(
                "category_bundle_no_text", str(category_message)
            )
            hint = f" (raw saved to {dump_path})" if dump_path else ""
            raise ValueError(f"Category bundle response missing text output{hint}")
        try:
            result = json.loads(text_val)
        except json.JSONDecodeError as exc:
            dump_path = dump_failed_response("category_bundle_bad_json", text_val)
            hint = f" (raw saved to {dump_path})" if dump_path else ""
            raise ValueError(f"Category bundle response was not valid JSON{hint}") from exc
        if not isinstance(result, dict):
            raise ValueError(f"Category bundle returned non-dict JSON: {type(result)}")

        quotes, category_summaries = normalize_category_bundle(result)

        summary_template = load_template(
            resolve_config_path(cfg.rag.summary_prompt_file, cfg)
        )
        summary_msg = summary_template.render(
            detail_level=cfg.ui.detail_level,
            category_summaries=category_summaries,
        )
        summary_kwargs: Dict[str, Any] = {
            "model": cfg.anthropic.model,
            "max_tokens": cfg.ui.max_output_tokens,
            "messages": [{"role": "user", "content": summary_msg}],
            "thinking": thinking,
        }
        if effort:
            summary_kwargs["output_config"] = {"effort": effort}
        if progress_callback:
            progress_callback("Generating summary")
        async with client.messages.stream(**summary_kwargs) as stream:
            summary_message = await stream.get_final_message()
        _ensure_message_completed(summary_message, "Summary", cfg.ui.max_output_tokens)
        key_takeaways = _extract_text(summary_message)

        return {"key_takeaways": key_takeaways, "quotes": quotes}
