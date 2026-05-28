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
- RAG retrieval and summarization pipeline
- User-editable prompt templates (Jinja2)
- RGB highlight colors editable in config
- Optional output directory for generated artifacts
- Markdown summary injected as sticky note on page 1
- Color-coded highlights: contributions (Y), limitations (O), claims (B), evidence (G)
- Quote-match report with matched/skipped counts, pages, scores, and verbose layout-gap diagnostics

## Getting Started

### 1. Installation

Quick start (local):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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
paperflux --config config.yaml path/to/your.pdf
# or reuse prior quotes without extraction
paperflux --config config.yaml --quotes-file your_paper_quotes.json path/to/your.pdf
```

#### Options

- `--config`, `-c`: Path to configuration file (required)
- `--detail`, `-d`: Detail level (low/medium/high, overrides config)
- `--verbose`: Enable internal logs, per-quote match details, and layout-gap diagnostics
- `--output-dir`, `-o`: Directory where annotated PDFs and summaries will be saved
- `--progress/--no-progress`: Show or hide per-file progress updates (default: shown)
- `--quotes-file`: Path to JSON quotes file to annotate without rerunning extraction

## Customizing Prompts

PaperFlux uses three editable Jinja2 templates in the `prompts/` directory to control how the AI extracts and summarizes information:

### `rag_category_system_prompt.txt`

The system prompt that instructs the AI assistant on its role and behavior. It defines the extraction rules:
- Use RAG file_search to find relevant passages
- Extract near-verbatim quotations with accurate page numbers
- Avoid including section/table/figure references in quotes
- Return structured JSON without code fences

This is the "personality" of the extraction assistant.

### `rag_category_prompt.j2`

The user prompt template for category extraction. It:
- Lists all extraction categories (contributions, limitations, claims, evidence) with their descriptions
- Specifies the JSON output format with categories, quotes, pages, and category summaries
- Gets rendered with variables from `config.yaml` (categories list)

Edit `config.yaml` (under `extraction_categories`) to change what categories are extracted. Edit this template to modify the output structure or instructions.

### `rag_summary_prompt.j2`

The prompt template for generating the final Markdown summary. It:
- Receives all category summaries as input
- Uses the detail level (low/medium/high) to control summary length
- Produces a cohesive narrative summary integrating all categories

Edit this to change how the final summary is structured or what information it emphasizes.

All three templates can be customized without modifying any Python code—just edit the files in `prompts/` and rerun PaperFlux.

## OpenAI Model and Retrieval Settings

The default model is `gpt-5.4-mini`, which supports the Responses API, structured outputs, and file search. You can change it in `config.yaml`:

```yaml
openai:
  model: "gpt-5.4-mini"
```

File-search retrieval can also be tuned without changing Python code:

```yaml
ui:
  reasoning_effort: "medium"
  max_output_tokens: 32768

matching:
  min_similarity: 0.88
  max_window_tokens: 80

rag:
  max_num_results:        # leave empty to let OpenAI choose
  max_quotes_per_category: 6
  include_search_results: false
  vector_store_expires_after_days: 1
```

Use `max_num_results` to cap retrieved passages when latency or cost matters. Use `max_quotes_per_category` to keep the structured JSON response bounded. The local highlighter uses `matching.min_similarity` and `matching.max_window_tokens` to align returned quotes to real PDF word spans; raising the similarity threshold improves precision, while lowering it improves recall. Enabling `include_search_results` is useful for debugging retrieval, but it increases the response payload.

The highlighter first tries exact and fuzzy contiguous matching. If those fail, it can fall back to `layout-gap` matching for quotes split by page layout interruptions such as tables, figures, captions, or column breaks. Layout-gap matches highlight only the words that belong to the quote and skip intervening layout artifacts.

## Brief History

- v1.0.20250525: Initial version using local text extraction, LLM-based processing of extracted text, and fuzzy quote matching for PDF annotation.
- v2.0.20250625: Migrated to OpenAI Responses API, adopted file_search with vector stores.
- v2.1.20251115: Added GPT-5 support, and bundled multi-category extraction into a single structured call.
- v2.2.20251115: Added n-gram quote matching for improved PDF annotation.
- v3.0.20251116: Added token-based quote matching for improved PDF annotation. Removed fuzzy- and n-gram-based matching.
- v3.1.20251116: Introduced a `--quotes-file` flow to re-annotate quickly without re-running extraction.
- v3.2.20260527: Updated default model to GPT-5.4 mini and added file-search tuning options.
- v3.3.20260527: Added local quote span alignment for more accurate PDF highlights.
- v3.4.20260528: Added layout-gap quote matching for quotes split by tables, figures, captions, or column breaks; added quote-match reports, concise default CLI output with verbose diagnostics, package metadata, and CLI validation improvements.

## Contributing

Contributions are very welcome! Whether you’re fixing a bug, improving extraction accuracy, or adding a new workflow, here’s how to get started:

1. Fork the repository and create a feature branch.
2. Set up the project and run tests locally.
3. Open a PR describing the change and its impact.

Issues, ideas, or questions? Please open an issue or start a discussion—feedback helps shape the roadmap.

---

For more information about configuration options and advanced features, see `config.yaml` in the repository.
