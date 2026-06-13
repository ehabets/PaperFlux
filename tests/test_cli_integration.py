import json
from types import SimpleNamespace

import fitz
from typer.testing import CliRunner

from paperflux import cli


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


def _write_anthropic_config(root):
    prompts_dir = root / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "category.j2").write_text(
        "Extract {{ max_quotes_per_category }} quote for "
        "{% for cat in categories %}{{ cat.name }}{% endfor %}.",
        encoding="utf-8",
    )
    (prompts_dir / "system_anthropic.txt").write_text(
        "Read the attached paper.", encoding="utf-8"
    )
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
provider: "anthropic"

anthropic:
  api_key: "test-key"
  model: "claude-test"

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
  category_system_prompt_file_anthropic: "prompts/system_anthropic.txt"
  summary_prompt_file: "prompts/summary.j2"
  max_quotes_per_category: 1
""",
        encoding="utf-8",
    )
    return config_path


def test_cli_full_pipeline_with_mocked_anthropic_and_tiny_pdf(tmp_path, monkeypatch):
    app_root = tmp_path / "app"
    app_root.mkdir()
    config_path = _write_anthropic_config(app_root)
    pdf_path = app_root / "paper.pdf"
    output_dir = app_root / "out"
    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    _write_tiny_pdf(pdf_path)

    stream_requests = []

    class FakeStream:
        def __init__(self, message):
            self._message = message

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_final_message(self):
            return self._message

    class FakeMessages:
        def stream(self, **kwargs):
            stream_requests.append(kwargs)
            output_config = kwargs.get("output_config") or {}
            if output_config.get("format"):
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
                text = json.dumps(payload)
            else:
                text = "The paper introduces a reliable method."
            message = SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=text)],
            )
            return FakeStream(message)

    class FakeAsyncAnthropic:
        def __init__(self, *, api_key):
            self.api_key = api_key
            self.messages = FakeMessages()

    monkeypatch.setattr(
        "paperflux.providers.anthropic_provider.AsyncAnthropic", FakeAsyncAnthropic
    )
    monkeypatch.chdir(outside_cwd)

    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            str(pdf_path),
        ],
    )

    assert result.exit_code == 0, result.output
    # First call extracts quotes (json_schema format + PDF document block).
    category_request = stream_requests[0]
    assert category_request["output_config"]["format"]["type"] == "json_schema"
    doc_blocks = [
        block
        for block in category_request["messages"][0]["content"]
        if block.get("type") == "document"
    ]
    assert doc_blocks and doc_blocks[0]["source"]["media_type"] == "application/pdf"
    assert "Extracting quotes with Claude" in result.output
    assert "Generating summary" in result.output
    assert "Creating temporary vector store" not in result.output

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

    monkeypatch.setattr("paperflux.providers.openai_provider.OpenAI", FakeOpenAI)
    monkeypatch.setattr("paperflux.providers.openai_provider.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.chdir(outside_cwd)

    result = CliRunner().invoke(
        cli.app,
        [
            "run",
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
            "run",
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


def test_cli_init_writes_config_and_prompt_templates(tmp_path):
    target_dir = tmp_path / "paperflux-project"

    result = CliRunner().invoke(cli.app, ["init", str(target_dir)])

    assert result.exit_code == 0, result.output
    assert f"Initialized PaperFlux project in: {target_dir}" in result.output
    config_path = target_dir / "config.yaml"
    assert config_path.exists()
    assert (target_dir / "prompts" / "rag_category_prompt.j2").exists()
    assert (target_dir / "prompts" / "rag_category_system_prompt.txt").exists()
    assert (target_dir / "prompts" / "rag_summary_prompt.j2").exists()
    assert 'api_key: "ENV:PAPERFLUX_OPENAI_API_KEY"' in config_path.read_text(
        encoding="utf-8"
    )


def test_cli_init_refuses_to_overwrite_without_force(tmp_path):
    target_dir = tmp_path / "paperflux-project"
    target_dir.mkdir()
    config_path = target_dir / "config.yaml"
    config_path.write_text("custom: true\n", encoding="utf-8")

    result = CliRunner().invoke(cli.app, ["init", str(target_dir)])

    assert result.exit_code == 1
    assert "Refusing to overwrite existing files:" in result.output
    assert config_path.read_text(encoding="utf-8") == "custom: true\n"

    force_result = CliRunner().invoke(cli.app, ["init", str(target_dir), "--force"])

    assert force_result.exit_code == 0, force_result.output
    assert 'api_key: "ENV:PAPERFLUX_OPENAI_API_KEY"' in config_path.read_text(
        encoding="utf-8"
    )


def test_console_entrypoint_preserves_legacy_run_invocation():
    assert cli._entrypoint_args(["--config", "config.yaml", "paper.pdf"]) == [
        "run",
        "--config",
        "config.yaml",
        "paper.pdf",
    ]
    assert cli._entrypoint_args(["init"]) == ["init"]
    assert cli._entrypoint_args(["--help"]) == ["--help"]
    assert cli._entrypoint_args(["--version"]) == ["--version"]
    assert cli._entrypoint_args(["-V"]) == ["-V"]


def test_version_flag_reports_package_version():
    from paperflux import __version__

    for flag in ("--version", "-V"):
        result = CliRunner().invoke(cli.app, [flag])
        assert result.exit_code == 0, result.output
        assert result.output.strip() == f"PaperFlux {__version__}"


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
            "run",
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
            "run",
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
            "run",
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
