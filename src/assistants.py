import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from openai import OpenAI, AsyncOpenAI
from .config import Config
from jinja2 import Environment, FileSystemLoader


def _load_template(template_path: str):
    """
    Load Jinja2 template given its filesystem path.
    """
    env = Environment(loader=FileSystemLoader("."))
    return env.get_template(template_path)


def _load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def _build_text_payload(format_payload: dict, verbosity: str) -> dict:
    return {"format": format_payload, "verbosity": verbosity}


def _multi_category_schema() -> dict:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "categories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "quotes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "text": {"type": "string"},
                                    "pages": {
                                        "type": "array",
                                        "items": {"type": "integer", "minimum": 1},
                                    },
                                },
                                "required": ["text", "pages"],
                            },
                        },
                        "category_summary": {"type": "string"},
                    },
                    "required": ["name", "quotes", "category_summary"],
                },
            }
        },
        "required": ["categories"],
    }
    return schema


def _reasoning_payload(effort: str) -> Dict[str, str]:
    return {"effort": effort}


def _dump_failed_response(label: str, content: str) -> Optional[Path]:
    """
    Persist the raw response body when JSON parsing fails so users can inspect it.
    """
    try:
        dump_dir = Path(".paperflux")
        dump_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)[:40]
        path = dump_dir / f"failed_response_{safe_label}_{timestamp}.txt"
        path.write_text(content, encoding="utf-8")
        return path
    except Exception:
        return None


def _ensure_response_completed(resp: Any, context: str, max_output_tokens: int) -> None:
    """
    Raise a descriptive error if the Responses API indicates the run was truncated.
    """
    status = getattr(resp, "status", None)
    if status and status != "completed":
        incomplete = getattr(resp, "incomplete_details", None)
        reason = getattr(incomplete, "reason", None)
        raise ValueError(
            f"{context} response ended with status='{status}' (reason='{reason}'). "
            f"Consider increasing ui.max_output_tokens (currently {max_output_tokens}) "
            "or lowering detail_level / category count."
        )


def _extract_response_text(resp) -> str:
    """
    Best-effort extraction of plain text from Responses API result.
    Prefers resp.output_text when available, with fallbacks for structured outputs.
    """
    # Newer SDKs expose output_text
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    # Fallback: iterate output structure
    try:
        output = getattr(resp, "output", None)
        if output and isinstance(output, list):
            chunks = []
            for item in output:
                # message/content/output_text structure
                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) == "output_text":
                            t = getattr(c, "text", "")
                            if t:
                                chunks.append(t)
            if chunks:
                return "".join(chunks)
    except (AttributeError, TypeError, ValueError):  # fallback for unknown SDK shapes
        pass

    # Last resort: stringify
    return str(resp)


def _extract_parsed_json(resp: Any) -> Optional[dict]:
    """
    Try to extract structured output when response_format=json_schema is used.
    """
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


async def analyze_pdf(path: Path, cfg: Config) -> dict:
    """
    Use OpenAI Responses API with built-in file_search tool to retrieve per-category evidence
    and synthesize a global summary. Retrieval happens server-side with tool_choice='auto'.
    - Creates a vector store
    - Uploads the PDF
    - For each category, runs responses.create with tools=[file_search] and tool_choice='auto'
      so the model can issue follow-up file_search calls (re-query, re-rank) automatically.
    - Executes category requests in parallel.
    - Generates a final markdown summary from category summaries.
    """
    # Sync client used for vector store creation/upload (simpler surface)
    client_sync = OpenAI(api_key=cfg.openai.api_key)

    # 1) Create vector store and upload PDF
    vector_store = client_sync.vector_stores.create(name="PaperFlux Vector Store")
    with open(path, "rb") as f:
        client_sync.vector_stores.files.upload_and_poll(
            vector_store_id=vector_store.id,
            file=f
        )

    categories = cfg.extraction_categories.categories  # Dict[name, description]

    # 2) Bundle categories into a single Responses job with file_search tool
    client_async = AsyncOpenAI(api_key=cfg.openai.api_key)
    category_template = _load_template(cfg.rag.category_prompt_file)
    category_system_prompt = _load_text_file(cfg.rag.category_system_prompt_file)

    category_entries = [
        {"name": cat, "description": desc}
        for cat, desc in categories.items()
    ]
    user_msg = category_template.render(categories=category_entries)

    tools = [{
        "type": "file_search",
        "vector_store_ids": [vector_store.id],
    }]

    schema = _multi_category_schema()
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
    kwargs["reasoning"] = reasoning_payload
    multi_resp = await client_async.responses.create(**kwargs)
    _ensure_response_completed(multi_resp, "Category bundle", cfg.ui.max_output_tokens)

    result = _extract_parsed_json(multi_resp)
    if result is None:
        text_val = _extract_response_text(multi_resp)
        if not text_val or not text_val.strip():
            dump_path = _dump_failed_response(
                "category_bundle_no_parsed_json",
                getattr(multi_resp, "output_text", "") or str(multi_resp),
            )
            hint = f" (raw saved to {dump_path})" if dump_path else ""
            raise ValueError(f"Category bundle response missing parsed JSON{hint}")
        result = json.loads(text_val)
    if not isinstance(result, dict):
        raise ValueError(f"Category bundle returned non-dict JSON: {type(result)}")

    bundle_list = result.get("categories")
    if not isinstance(bundle_list, list):
        raise ValueError("Category bundle JSON missing 'categories' array")

    quotes: Dict[str, list] = {}
    category_summaries: Dict[str, str] = {}
    for entry in bundle_list:
        if not isinstance(entry, dict):
            continue
        category = entry.get("name") or entry.get("category")
        if not isinstance(category, str):
            continue
        cat_data = entry
        raw_quotes = (
            cat_data.get("quotes")
            or cat_data.get("evidence")
            or cat_data.get("quote_list")
            or []
        )
        normalised_quotes = []
        for item in raw_quotes:
            if isinstance(item, dict):
                text_val = item.get("text") or item.get("quote") or item.get("content")
                if not isinstance(text_val, str) or not text_val.strip():
                    continue
                pages_val = item.get("pages") or item.get("page")
                if isinstance(pages_val, int):
                    pages = [pages_val]
                elif isinstance(pages_val, list):
                    pages = [p for p in pages_val if isinstance(p, int) and p > 0]
                else:
                    pages = []
                normalised_quotes.append({
                    "text": text_val.strip(),
                    "pages": pages,
                })
            elif isinstance(item, str) and item.strip():
                normalised_quotes.append({"text": item.strip(), "pages": []})
        quotes[category] = normalised_quotes
        summary_val = cat_data.get("category_summary") or cat_data.get("summary") or ""
        category_summaries[category] = summary_val

    # 3) Synthesize global summary (no tools required)
    summary_template = _load_template(cfg.rag.summary_prompt_file)
    summary_msg = summary_template.render(
        detail_level=cfg.ui.detail_level,
        category_summaries=category_summaries
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
    summary_resp = await client_async.responses.create(**summary_kwargs)
    _ensure_response_completed(summary_resp, "Summary", cfg.ui.max_output_tokens)
    key_takeaways = _extract_response_text(summary_resp)

    return {"key_takeaways": key_takeaways, "quotes": quotes}
