import json
from types import SimpleNamespace

import fitz
from typer.testing import CliRunner

from src import cli


def _write_tiny_pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This paper introduces a reliable method.", fontsize=12)
    doc.save(path)
    doc.close()


def _write_config(root):
    prompts_dir = root / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "category.j2").write_text(
        "Extract {{ max_quotes_per_category }} quote for "
        "{% for cat in categories %}{{ cat.name }}{% endfor %}.",
        encoding="utf-8",
    )
    (prompts_dir / "system.txt").write_text("Use the attached paper.", encoding="utf-8")
    (prompts_dir / "summary.j2").write_text(
        "{{ detail_level }} summary: "
        "{% for category, summary in category_summaries.items() %}"
        "{{ category }}={{ summary }}"
        "{% endfor %}",
        encoding="utf-8",
    )

    config_path = root / "config.yaml"
    config_path.write_text(
        """
openai:
  api_key: "test-key"
  model: "test-model"

ui:
  detail_level: "low"
  reasoning_effort: "low"
  max_output_tokens: 2048
  highlight_colors:
    contributions: [1.0, 1.0, 0.0]

extraction_categories:
  categories:
    contributions: "Important contributions."

matching:
  min_similarity: 0.88
  max_window_tokens: 80

rag:
  category_prompt_file: "prompts/category.j2"
  category_system_prompt_file: "prompts/system.txt"
  summary_prompt_file: "prompts/summary.j2"
  max_quotes_per_category: 1
""",
        encoding="utf-8",
    )
    return config_path


def test_cli_full_pipeline_with_mocked_openai_and_tiny_pdf(tmp_path, monkeypatch):
    app_root = tmp_path / "app"
    app_root.mkdir()
    config_path = _write_config(app_root)
    pdf_path = app_root / "paper.pdf"
    output_dir = app_root / "out"
    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    _write_tiny_pdf(pdf_path)

    created_vector_stores = []
    deleted_vector_stores = []
    uploaded_vector_stores = []
    response_requests = []

    class FakeFiles:
        def upload_and_poll(self, *, vector_store_id, file):
            uploaded_vector_stores.append(vector_store_id)
            file.read(1)
            return SimpleNamespace(id="vs_file_test")

    class FakeVectorStores:
        def __init__(self):
            self.files = FakeFiles()

        def create(self, **kwargs):
            created_vector_stores.append(kwargs)
            return SimpleNamespace(id="vs_test")

        def delete(self, *, vector_store_id):
            deleted_vector_stores.append(vector_store_id)
            return SimpleNamespace(id=vector_store_id, deleted=True)

    class FakeOpenAI:
        def __init__(self, *, api_key):
            self.api_key = api_key
            self.vector_stores = FakeVectorStores()

    class FakeResponses:
        async def create(self, **kwargs):
            response_requests.append(kwargs)
            if kwargs.get("tools"):
                payload = {
                    "categories": [
                        {
                            "name": "contributions",
                            "quotes": [
                                {
                                    "text": "This paper introduces a reliable method.",
                                    "pages": [1],
                                    "prefix": "",
                                    "suffix": "",
                                }
                            ],
                            "category_summary": "The paper introduces a reliable method.",
                        }
                    ]
                }
                return SimpleNamespace(status="completed", output_text=json.dumps(payload))
            return SimpleNamespace(
                status="completed",
                output_text="The paper introduces a reliable method.",
            )

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key):
            self.api_key = api_key
            self.responses = FakeResponses()

    monkeypatch.setattr("src.assistants.OpenAI", FakeOpenAI)
    monkeypatch.setattr("src.assistants.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.chdir(outside_cwd)

    result = CliRunner().invoke(
        cli.app,
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            str(pdf_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert created_vector_stores
    assert uploaded_vector_stores == ["vs_test"]
    assert deleted_vector_stores == ["vs_test"]
    assert response_requests[0]["tools"][0]["vector_store_ids"] == ["vs_test"]
    assert "contributions" in response_requests[0]["input"][1]["content"]
    assert "Input" in result.output
    assert "Processing" in result.output
    assert "[1/1] Processing paper.pdf" in result.output
    assert "Creating temporary vector store" in result.output
    assert "Uploading and indexing paper.pdf" in result.output
    assert "Extracting quotes with OpenAI" in result.output
    assert "Generating summary" in result.output
    assert "Cleaning up temporary vector store" in result.output
    assert "Annotating PDF and matching quotes" in result.output
    assert "Writing markdown, quotes, and match report" in result.output
    assert "Outputs" in result.output
    assert "Quote Matches" in result.output
    assert "- Summary: 1/1 matched, 0 skipped" in result.output
    assert "Matched quotes:" not in result.output
    assert "contributions #1: p. 1, exact, score" not in result.output

    report_path = output_dir / "paper_quote_matches.json"
    assert (output_dir / "paper_annotated.pdf").exists()
    assert (output_dir / "paper_summary.md").exists()
    assert (output_dir / "paper_quotes.json").exists()
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["matched"] == 1
    assert report["skipped"] == 0
    assert report["records"][0]["page"] == 1
    assert report["records"][0]["score"] >= 0.88


def test_cli_no_progress_disables_stage_callback(tmp_path, monkeypatch):
    app_root = tmp_path / "app"
    app_root.mkdir()
    config_path = _write_config(app_root)
    pdf_path = app_root / "paper.pdf"
    output_dir = app_root / "out"
    _write_tiny_pdf(pdf_path)
    captured = {}

    async def fake_batch_process(
        pdf_paths,
        cfg,
        verbose,
        output_dir=None,
        show_progress=True,
        progress_callback=None,
    ):
        captured["show_progress"] = show_progress
        captured["progress_callback"] = progress_callback
        target_dir = output_dir or pdf_paths[0].parent
        target_dir.mkdir(parents=True, exist_ok=True)
        pdf_out = target_dir / "paper_annotated.pdf"
        md_out = target_dir / "paper_summary.md"
        quotes_out = target_dir / "paper_quotes.json"
        match_report_out = target_dir / "paper_quote_matches.json"
        pdf_out.write_bytes(b"%PDF-1.4\n")
        md_out.write_text("# Summary\n", encoding="utf-8")
        quotes_out.write_text('{"quotes": {}}', encoding="utf-8")
        match_report_out.write_text(
            json.dumps({"total": 0, "matched": 0, "skipped": 0, "records": []}),
            encoding="utf-8",
        )
        return [(pdf_out, md_out, quotes_out, match_report_out)]

    monkeypatch.setattr(cli, "batch_process", fake_batch_process)

    result = CliRunner().invoke(
        cli.app,
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--no-progress",
            str(pdf_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["show_progress"] is False
    assert captured["progress_callback"] is None
    assert "Creating temporary vector store" not in result.output


def test_quote_match_report_prints_details_only_when_verbose(tmp_path, capsys):
    report_path = tmp_path / "paper_quote_matches.json"
    report_path.write_text(
        json.dumps(
            {
                "total": 2,
                "matched": 1,
                "skipped": 1,
                "records": [
                    {
                        "category": "limitations",
                        "quote_index": 1,
                        "text": "split quote",
                        "matched": True,
                        "page": 4,
                        "score": 0.94,
                        "method": "layout-gap",
                        "segments": 4,
                        "matched_text": "split quote",
                        "skipped_reason": None,
                    },
                    {
                        "category": "claims",
                        "quote_index": 2,
                        "text": "missing quote",
                        "matched": False,
                        "page": None,
                        "score": None,
                        "method": None,
                        "segments": 0,
                        "matched_text": "",
                        "skipped_reason": "not found",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    cli._echo_quote_match_report(report_path, verbose=False)
    default_output = capsys.readouterr().out
    assert "Quote Matches" in default_output
    assert "- Summary: 1/2 matched, 1 skipped" in default_output
    assert "- Methods: layout-gap 1" in default_output
    assert "Layout-gap matches:" not in default_output
    assert "- limitations #1: p. 4, score 0.940, 4 segments" not in default_output
    assert "Skipped quotes:" in default_output
    assert "missing quote" in default_output
    assert "Matched quotes:" not in default_output
    assert "Run with --verbose" not in default_output

    cli._echo_quote_match_report(report_path, verbose=True)
    verbose_output = capsys.readouterr().out
    assert "Layout-gap matches:" in verbose_output
    assert "- limitations #1: p. 4, score 0.940, 4 segments" in verbose_output
    assert "Matched quotes:" in verbose_output
    assert "- limitations #1: p. 4, layout-gap, score 0.940, segments 4" in verbose_output


def test_cli_rejects_invalid_detail_override(tmp_path):
    app_root = tmp_path / "app"
    app_root.mkdir()
    config_path = _write_config(app_root)
    pdf_path = app_root / "paper.pdf"
    _write_tiny_pdf(pdf_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "--config",
            str(config_path),
            "--detail",
            "nonsense",
            str(pdf_path),
        ],
    )

    assert result.exit_code == 1
    assert "Invalid CLI override" in result.output


def test_cli_processing_failure_exits_nonzero(tmp_path, monkeypatch):
    app_root = tmp_path / "app"
    app_root.mkdir()
    config_path = _write_config(app_root)
    pdf_path = app_root / "paper.pdf"
    _write_tiny_pdf(pdf_path)

    async def failing_batch_process(*args, **kwargs):
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr(cli, "batch_process", failing_batch_process)

    result = CliRunner().invoke(
        cli.app,
        [
            "--config",
            str(config_path),
            str(pdf_path),
        ],
    )

    assert result.exit_code == 1
    assert "Error during processing: pipeline failed" in result.output


def test_cli_rejects_quotes_file_with_multiple_pdfs(tmp_path):
    app_root = tmp_path / "app"
    app_root.mkdir()
    config_path = _write_config(app_root)
    first_pdf = app_root / "first.pdf"
    second_pdf = app_root / "second.pdf"
    quotes_path = app_root / "quotes.json"
    _write_tiny_pdf(first_pdf)
    _write_tiny_pdf(second_pdf)
    quotes_path.write_text('{"key_takeaways": "", "quotes": {}}', encoding="utf-8")

    result = CliRunner().invoke(
        cli.app,
        [
            "--config",
            str(config_path),
            "--quotes-file",
            str(quotes_path),
            str(first_pdf),
            str(second_pdf),
        ],
    )

    assert result.exit_code == 1
    assert "--quotes-file can be used with exactly one PDF." in result.output
