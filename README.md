# PaperFlux

AI-powered PDF annotation for research papers. PaperFlux extracts exact quotations, organizes them by category (contributions, limitations, claims, evidence), and annotates your PDFs with precise highlights. It works with either OpenAI or Anthropic (Claude) models.

## Quick Start

### 1. Installation

Install the latest release from PyPI:

```bash
python -m pip install paperflux
```

For local development from a cloned repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Set API Key

PaperFlux uses OpenAI by default. Set the key for the provider you plan to use:

```bash
# OpenAI (default)
export PAPERFLUX_OPENAI_API_KEY="sk-your-key"

# Anthropic (when provider: "anthropic")
export PAPERFLUX_ANTHROPIC_API_KEY="sk-ant-your-key"
```

Select the backend with the `provider` key in `config.yaml` (`"openai"` or `"anthropic"`).

### 3. Run

```bash
paperflux init
paperflux --config config.yaml path/to/paper.pdf
```

## Features

- Pluggable LLM backend: OpenAI or Anthropic (Claude)
- Batch processing: `*.pdf`
- Three detail levels (low/medium/high)
- RAG-based extraction with exact quotes
- Color-coded highlights by category
- Markdown summary with sticky note
- Quote-match report with matched/skipped counts and scores
- Layout-aware quote matching across column, table, figure, and caption interruptions
- Stage-level CLI progress during extraction and annotation
- Configurable prompts and colors

## Documentation

For detailed setup, configuration options, and advanced usage, see the [full documentation](https://ehabets.github.io/PaperFlux/).

## Contributing

Contributions welcome! Fork the repo, create a feature branch, and open a PR.

## License

See [LICENSE](LICENSE) for details.
