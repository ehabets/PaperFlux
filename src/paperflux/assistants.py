"""Public LLM entry point for PaperFlux.

Dispatches to the configured provider (``cfg.provider``). Each provider returns
the canonical ``{"key_takeaways": str, "quotes": Dict[str, list]}`` shape, so the
rest of the pipeline is provider-agnostic.
"""

import logging
from pathlib import Path
from typing import Optional

from .config import Config
from .providers import ProgressCallback, get_provider

logger = logging.getLogger(__name__)


async def analyze_pdf(
    path: Path,
    cfg: Config,
    progress_callback: Optional[ProgressCallback] = None,
) -> dict:
    """Analyze a PDF with the configured provider and return quotes + summary."""
    provider = get_provider(cfg.provider)
    return await provider.analyze_pdf(path, cfg, progress_callback=progress_callback)
