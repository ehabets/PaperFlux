"""Shared building blocks for PaperFlux LLM providers.

A provider turns a PDF into the pipeline's canonical result shape,
``{"key_takeaways": str, "quotes": Dict[str, list]}``. The helpers here are
provider-neutral (prompt/template loading, the extraction JSON schema, and the
bundle-normalisation routine) so each concrete provider only implements the
parts that differ between SDKs.
"""

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

try:  # pragma: no cover - typing-only convenience
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore

from jinja2 import Environment, FileSystemLoader

from ..config import Config

ProgressCallback = Callable[[str], None]


class LLMProvider(Protocol):
    """Interface every backend implements.

    Implementations return ``{"key_takeaways": str, "quotes": Dict[str, list]}``
    so the rest of the pipeline stays provider-agnostic.
    """

    async def analyze_pdf(
        self,
        path: Path,
        cfg: Config,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> dict:
        ...


def resolve_config_path(path: Union[str, Path], cfg: Config) -> Path:
    """Resolve a config-relative path against the loaded config's directory."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    config_dir = getattr(cfg, "_config_dir", None)
    if config_dir is not None:
        return config_dir / candidate
    return candidate


def load_template(template_path: Union[str, Path]):
    """Load a Jinja2 template given its filesystem path."""
    template_path = Path(template_path)
    env = Environment(loader=FileSystemLoader(str(template_path.parent or Path("."))))
    return env.get_template(template_path.name)


def load_text_file(path: Union[str, Path]) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def multi_category_schema(max_quotes_per_category: int) -> dict:
    """JSON schema for the bundled per-category quote extraction response."""
    return {
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
                            "maxItems": max_quotes_per_category,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "text": {"type": "string"},
                                    "pages": {
                                        "type": "array",
                                        "items": {"type": "integer", "minimum": 1},
                                    },
                                    "prefix": {"type": "string"},
                                    "suffix": {"type": "string"},
                                },
                                "required": ["text", "pages", "prefix", "suffix"],
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


def normalize_category_bundle(
    result: dict,
) -> Tuple[Dict[str, list], Dict[str, str]]:
    """Normalise a parsed category bundle into ``(quotes, category_summaries)``.

    Tolerant of provider phrasing differences (alternate key names, scalar vs.
    list pages, bare-string quotes) so both backends feed the same downstream
    quote-matching pipeline.
    """
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
        raw_quotes = (
            entry.get("quotes")
            or entry.get("evidence")
            or entry.get("quote_list")
            or []
        )
        normalised_quotes: List[dict] = []
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
                prefix_val = item.get("prefix") or item.get("context_before") or ""
                suffix_val = item.get("suffix") or item.get("context_after") or ""
                normalised_quotes.append({
                    "text": text_val.strip(),
                    "pages": pages,
                    "prefix": prefix_val.strip() if isinstance(prefix_val, str) else "",
                    "suffix": suffix_val.strip() if isinstance(suffix_val, str) else "",
                })
            elif isinstance(item, str) and item.strip():
                normalised_quotes.append({
                    "text": item.strip(),
                    "pages": [],
                    "prefix": "",
                    "suffix": "",
                })
        quotes[category] = normalised_quotes
        summary_val = entry.get("category_summary") or entry.get("summary") or ""
        category_summaries[category] = summary_val

    return quotes, category_summaries


def dump_failed_response(label: str, content: str) -> Optional[Path]:
    """Persist a raw response body when JSON parsing fails, for inspection."""
    try:
        dump_dir = Path(".paperflux")
        dump_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        safe_label = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in label
        )[:40]
        path = dump_dir / f"failed_response_{safe_label}_{timestamp}.txt"
        path.write_text(content, encoding="utf-8")
        return path
    except Exception:
        return None
