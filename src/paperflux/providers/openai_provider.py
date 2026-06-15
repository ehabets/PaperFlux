"""OpenAI backend: Responses API + server-side vector store / file_search RAG."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI, AsyncOpenAI

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


def _build_text_payload(format_payload: dict, verbosity: str) -> dict:
    """Build the ``text`` parameter dict for a Responses API call."""
    return {"format": format_payload, "verbosity": verbosity}


def _reasoning_payload(effort: str) -> Dict[str, str]:
    """Build the ``reasoning`` parameter dict for a Responses API call."""
    return {"effort": effort}


def _ensure_response_completed(resp: Any, context: str, max_output_tokens: int) -> None:
    """Raise a descriptive error if the Responses API indicates truncation."""
    status = getattr(resp, "status", None)
    if status and status != "completed":
        incomplete = getattr(resp, "incomplete_details", None)
        reason = getattr(incomplete, "reason", None)
        raise ValueError(
            f"{context} response ended with status='{status}' (reason='{reason}'). "
            f"Consider increasing ui.max_output_tokens (currently {max_output_tokens}) "
            "or lowering ui.reasoning_effort, detail_level, category count, or "
            "rag.max_quotes_per_category."
        )


def _extract_response_text(resp) -> str:
    """Best-effort extraction of plain text from a Responses API result."""
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    try:
        output = getattr(resp, "output", None)
        if output and isinstance(output, list):
            chunks = []
            for item in output:
                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) == "output_text":
                            t = getattr(c, "text", "")
                            if t:
                                chunks.append(t)
            if chunks:
                return "".join(chunks)
    except (AttributeError, TypeError, ValueError):  # unknown SDK shapes
        pass

    return str(resp)


def _extract_parsed_json(resp: Any) -> Optional[dict]:
    """Extract structured output when response_format=json_schema is used."""
    parsed = getattr(resp, "output_parsed", None)
    if parsed:
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
    output = getattr(resp, "output", None)
    if isinstance(output, list):
        for item in output:
            for c in getattr(item, "content", []) or []:
                maybe = getattr(c, "parsed", None)
                if isinstance(maybe, dict):
                    return maybe
    return None


class OpenAIProvider:
    """Use the OpenAI Responses API with the built-in file_search tool.

    Retrieval happens server-side: a temporary vector store is created, the PDF
    is uploaded and indexed, one bundled category-extraction response runs with
    file_search enabled, then a markdown summary is synthesized.
    """

    async def analyze_pdf(
        self,
        path: Path,
        cfg: Config,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> dict:
        """Analyze a PDF and return extracted quotes and a summary.

        Uploads the PDF to a temporary OpenAI vector store, runs a bundled
        category-extraction call with file_search, then synthesizes a global
        summary. The vector store is deleted in a ``finally`` block regardless
        of outcome.

        Args:
            path: Path to the PDF file to analyze.
            cfg: Loaded PaperFlux configuration.
            progress_callback: Optional callable invoked with a status string
                at each major pipeline stage.

        Returns:
            A dict with keys ``"key_takeaways"`` (str) and ``"quotes"``
            (Dict[category_name, list]).
        """
        client_sync = OpenAI(api_key=cfg.openai.api_key)
        vector_store_id: Optional[str] = None

        try:
            # 1) Create vector store and upload PDF
            if progress_callback:
                progress_callback("Creating temporary vector store")
            vector_store = client_sync.vector_stores.create(
                name="PaperFlux Vector Store",
                expires_after={
                    "anchor": "last_active_at",
                    "days": cfg.rag.vector_store_expires_after_days,
                },
            )
            vector_store_id = vector_store.id
            if progress_callback:
                progress_callback(f"Uploading and indexing {path.name}")
            with open(path, "rb") as f:
                client_sync.vector_stores.files.upload_and_poll(
                    vector_store_id=vector_store_id,
                    file=f,
                )

            categories = cfg.extraction_categories.categories  # Dict[name, description]

            # 2) Bundle categories into a single Responses job with file_search
            client_async = AsyncOpenAI(api_key=cfg.openai.api_key)
            category_template = load_template(
                resolve_config_path(cfg.rag.category_prompt_file, cfg)
            )
            category_system_prompt = load_text_file(
                resolve_config_path(cfg.rag.category_system_prompt_file, cfg)
            )

            category_entries = [
                {"name": cat, "description": desc}
                for cat, desc in categories.items()
            ]
            user_msg = category_template.render(
                categories=category_entries,
                max_quotes_per_category=cfg.rag.max_quotes_per_category,
            )

            file_search_tool = {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }
            if cfg.rag.max_num_results is not None:
                file_search_tool["max_num_results"] = cfg.rag.max_num_results
            tools = [file_search_tool]

            schema = multi_category_schema(cfg.rag.max_quotes_per_category)
            text_payload = _build_text_payload(
                {
                    "type": "json_schema",
                    "name": "multi_category_schema",
                    "schema": schema,
                    "strict": True,
                },
                cfg.ui.verbosity,
            )
            reasoning_payload = _reasoning_payload(cfg.ui.reasoning_effort)
            request_input = [
                {"role": "system", "content": category_system_prompt},
                {"role": "user", "content": user_msg},
            ]
            kwargs = {
                "model": cfg.openai.model,
                "input": request_input,
                "max_output_tokens": cfg.ui.max_output_tokens,
                "tools": tools,
                "text": text_payload,
                "tool_choice": {"type": "file_search"},
                "store": False,
            }
            if cfg.rag.include_search_results:
                kwargs["include"] = ["file_search_call.results"]
            kwargs["reasoning"] = reasoning_payload
            if progress_callback:
                progress_callback("Extracting quotes with OpenAI")
            multi_resp = await client_async.responses.create(**kwargs)
            _ensure_response_completed(
                multi_resp, "Category bundle", cfg.ui.max_output_tokens
            )

            result = _extract_parsed_json(multi_resp)
            if result is None:
                text_val = _extract_response_text(multi_resp)
                if not text_val or not text_val.strip():
                    dump_path = dump_failed_response(
                        "category_bundle_no_parsed_json",
                        getattr(multi_resp, "output_text", "") or str(multi_resp),
                    )
                    hint = f" (raw saved to {dump_path})" if dump_path else ""
                    raise ValueError(f"Category bundle response missing parsed JSON{hint}")
                result = json.loads(text_val)
            if not isinstance(result, dict):
                raise ValueError(f"Category bundle returned non-dict JSON: {type(result)}")

            quotes, category_summaries = normalize_category_bundle(result)

            # 3) Synthesize global summary (no tools required)
            summary_template = load_template(
                resolve_config_path(cfg.rag.summary_prompt_file, cfg)
            )
            summary_msg = summary_template.render(
                detail_level=cfg.ui.detail_level,
                category_summaries=category_summaries,
            )
            summary_text_payload = _build_text_payload({"type": "text"}, cfg.ui.verbosity)
            summary_reasoning = _reasoning_payload(cfg.ui.reasoning_effort)
            summary_kwargs = {
                "model": cfg.openai.model,
                "input": summary_msg,
                "max_output_tokens": cfg.ui.max_output_tokens,
                "text": summary_text_payload,
                "store": False,
            }
            summary_kwargs["reasoning"] = summary_reasoning
            if progress_callback:
                progress_callback("Generating summary")
            summary_resp = await client_async.responses.create(**summary_kwargs)
            _ensure_response_completed(summary_resp, "Summary", cfg.ui.max_output_tokens)
            key_takeaways = _extract_response_text(summary_resp)

            return {"key_takeaways": key_takeaways, "quotes": quotes}
        finally:
            if vector_store_id:
                try:
                    if progress_callback:
                        progress_callback("Cleaning up temporary vector store")
                    client_sync.vector_stores.delete(vector_store_id=vector_store_id)
                except Exception as exc:
                    logger.warning(
                        "Failed to delete vector store %s: %s", vector_store_id, exc
                    )
