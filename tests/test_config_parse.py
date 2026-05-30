import pytest

from paperflux.config import Config, load

def test_config_parse(tmp_path):
    # Create a sample config.yaml file
    config_content = """
openai:
  api_key: "testkey"
  model: "gpt-5.4-mini"

ui:
  detail_level: "medium"
  reasoning_effort: "low"
  max_output_tokens: 32768
  highlight_colors:
    contributions: [1.0, 1.0, 0.0]
    limitations:   [1.0, 0.6, 0.0]
    claims:        [0.2, 0.4, 1.0]
    evidence:      [0.0, 0.8, 0.3]

matching:
  min_similarity: 0.9
  max_window_tokens: 120

rag:
  category_prompt_file: "prompts/rag_category_prompt.j2"
  summary_prompt_file: "prompts/rag_summary_prompt.j2"
  max_num_results: 7
  max_quotes_per_category: 5
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    # Load config
    cfg = load(config_file)

    # Assert values
    assert isinstance(cfg, Config)
    assert cfg.openai.api_key == "testkey"
    assert cfg.openai.model == "gpt-5.4-mini"
    assert cfg.ui.detail_level == "medium"
    assert cfg.ui.reasoning_effort == "low"
    assert cfg.ui.max_output_tokens == 32768
    assert cfg.ui.highlight_colors["contributions"] == [1.0, 1.0, 0.0]
    assert cfg.matching.min_similarity == 0.9
    assert cfg.matching.max_window_tokens == 120
    assert cfg.rag.category_prompt_file == "prompts/rag_category_prompt.j2"
    assert cfg.rag.summary_prompt_file == "prompts/rag_summary_prompt.j2"
    assert cfg.rag.max_num_results == 7
    assert cfg.rag.max_quotes_per_category == 5


def test_missing_highlight_colors(tmp_path):
    config_content = """
openai:
  api_key: "testkey"
  model: "gpt-5.4-mini"

ui:
  detail_level: "medium"
  highlight_colors:
    contributions: [1.0, 1.0, 0.0]

extraction_categories:
  categories:
    contributions: "..."
    methodology: "..."

rag:
  category_prompt_file: "prompts/rag_category_prompt.j2"
  summary_prompt_file: "prompts/rag_summary_prompt.j2"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    with pytest.raises(ValueError) as excinfo:
        load(config_file)

    assert "Highlight colors missing" in str(excinfo.value)
