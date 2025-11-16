# PaperFlux

AI-powered PDF annotation for research papers. PaperFlux extracts exact quotations, organizes them by category (contributions, limitations, claims, evidence), and annotates your PDFs with precise highlights.

## Quick Start

### 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="sk-your-key"
```

### 3. Run

```bash
python -m src.cli --config config.yaml path/to/paper.pdf
```

## Features

- Batch processing: `*.pdf`
- Three detail levels (low/medium/high)
- RAG-based extraction with exact quotes
- Color-coded highlights by category
- Markdown summary with sticky note
- Configurable prompts and colors

## Documentation

For detailed setup, configuration options, and advanced usage, see the [full documentation](https://ehabets.github.io/PaperFlux/).

## Contributing

Contributions welcome! Fork the repo, create a feature branch, and open a PR.

## License

See [LICENSE](LICENSE) for details.
