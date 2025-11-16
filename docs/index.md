---
layout: default
title: PaperFlux
description: AI-powered PDF annotation for research papers
---

# PaperFlux

PaperFlux helps you read scientific papers faster by automatically extracting exact quotations, organizing them by category (e.g., contributions, limitations, claims, evidence), and annotating your PDFs with precise highlights. It also produces a concise, structured summary you can share or extend. The current version requires an OpenAI API key.

## The Idea

- Use Retrieval-Augmented Generation (RAG) to find the most relevant passages inside your paper via server-side file search.
- Ask the model to return structured results (JSON) with exact quotes and page numbers per category.
- Turn that into actionable artifacts: an annotated PDF with color-coded highlights and a clean Markdown summary.
- Keep everything reproducible: save the extracted quotes alongside your outputs so you can re-annotate without re-running extraction.

## Features

- Batch CLI: [options] *.pdf
- YAML config with LLMs, prompts, colors, defaults
- Three detail levels (low / medium / high)
- Assistants-driven RAG retrieval/summary pipeline
- User-editable prompt templates (Jinja2)
- RGB highlight colors editable in config
- Optional output directory for generated artifacts
- Markdown summary injected as sticky note on page 1
- Color-coded highlights: contributions (Y), limitations (O), claims (B), evidence (G)

## Getting Started

### 1. Installation

Quick start (local):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

### 2. Configure the API Key

PaperFlux requires an OpenAI API key. The key is read from an environment variable referenced in `config.yaml`:

- In `config.yaml`, the default is:

	```yaml
	openai:
		api_key: "ENV:OPENAI_API_KEY"
	```

	The `ENV:` prefix means PaperFlux will expand the environment variable `OPENAI_API_KEY` at runtime.

Choose one of the following setups:

- Temporary (current shell only):

	```bash
	export OPENAI_API_KEY="sk-your-key"
	```

- Persistent for zsh (macOS default):

	```bash
	echo 'export OPENAI_API_KEY="sk-your-key"' >> ~/.zshrc
	source ~/.zshrc
	```

- Using a `.env` file (auto-loaded):

	Create a file named `.env` in the repository root:

	```bash
	echo 'OPENAI_API_KEY=sk-your-key' > .env
	```

	PaperFlux loads `.env` automatically because it calls `load_dotenv()` during config load.

- Inline in `config.yaml` (not recommended for commits):

	```yaml
	openai:
		api_key: "sk-your-key"
	```

### 3. Usage

Run the CLI:

```bash
python -m src.cli --config config.yaml --verbose path/to/your.pdf
# or reuse prior quotes without extraction
python -m src.cli --config config.yaml --quotes-file your_paper_quotes.json path/to/your.pdf
```

#### Options

- `--config`, `-c`: Path to configuration file (required)
- `--detail`, `-d`: Detail level (low/medium/high, overrides config)
- `--verbose`: Enable verbose output
- `--output-dir`, `-o`: Directory where annotated PDFs and summaries will be saved
- `--progress/--no-progress`: Show or hide per-file progress updates (default: shown)
- `--quotes-file`: Path to JSON quotes file to annotate without rerunning extraction

## Brief History

- v1.0.20250525: Initial version using local text extraction, LLM-based processing of extracted text, and fuzzy quote matching for PDF annotation.
- v2.0.20250625: Migrated to OpenAI Responses API, adopted file_search with vector stores.
- v2.1.20251115: Added GPT-5 support, and bundled multi-category extraction into a single structured call.
- v2.2.20251115: Added n-gram quote matching for improved PDF annotation.
- v3.0.20251116: Added token-based quote matching for improved PDF annotation. Removed fuzzy- and n-gram-based matching.
- v3.1.20251116: Introduced a `--quotes-file` flow to re-annotate quickly without re-running extraction.

## Contributing

Contributions are very welcome! Whether you’re fixing a bug, improving extraction accuracy, or adding a new workflow, here’s how to get started:

1. Fork the repository and create a feature branch.
2. Set up the project and run tests locally.
3. Open a PR describing the change and its impact.

Issues, ideas, or questions? Please open an issue or start a discussion—feedback helps shape the roadmap.

---

For more information about configuration options and advanced features, see `config.yaml` in the repository.
