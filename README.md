# PaperFlux

AI-powered PDF annotation for research papers. PaperFlux extracts exact quotations, organizes them by category (contributions, limitations, claims, evidence), and annotates your PDFs with precise highlights.

## Quick Start

### 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="sk-your-key"
```

### 3. Run

```bash
paperflux --config config.yaml path/to/paper.pdf
```

## Features

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
