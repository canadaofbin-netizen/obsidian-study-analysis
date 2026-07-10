import os
import sys
import pytest
import yaml
from unittest.mock import patch, MagicMock

# Ensure scripts directory is in python path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_scripts_dir = os.path.abspath(os.path.join(_current_dir, "..", "scripts"))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from extract_facts import clean_and_format_markdown
from run_debate import main as run_debate_main

def test_clean_and_format_markdown_nested_code_blocks():
    """
    Adversarial test: Verify that clean_and_format_markdown does not truncate
    the note content when it contains nested code blocks (e.g. Python code blocks
    inside a note wrapped in a markdown code block).
    """
    input_text = """```markdown
---
tags:
  - fact_note
---
# Fact Note: Title

Here is a python script:
```python
print("Hello")
```
```"""
    output = clean_and_format_markdown(input_text)
    
    # Check that the Python code block is preserved and not truncated
    assert 'print("Hello")' in output
    assert '```python' in output


def test_clean_and_format_markdown_missing_boundary_bullet_points():
    """
    Adversarial test: Verify that if a closing boundary '---' is missing and
    the body starts with bullet points, they are not consumed as tags.
    """
    input_text = """---
tags:
- fact_note

- This is a body bullet list.
- And another one.
"""
    output = clean_and_format_markdown(input_text)
    
    # Parse the output's YAML frontmatter
    lines = output.splitlines()
    assert lines[0] == "---"
    closing_idx = -1
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_idx = idx
            break
    assert closing_idx != -1
    
    yaml_block = "\n".join(lines[1:closing_idx])
    parsed_yaml = yaml.safe_load(yaml_block)
    
    # The bullet points should NOT be tags!
    tags = parsed_yaml.get("tags", [])
    for tag in tags:
        assert "body bullet" not in tag.lower(), f"Bullet points was incorrectly parsed as tag: {tag}"


@patch('google.generativeai.GenerativeModel')
def test_debate_with_single_extracted_source_paper(mock_model_class, tmp_path):
    """
    Adversarial test: Verify that if the LLM transcript only references 1 source paper,
    run_debate either warns, errors, or fails, and does not silently generate a
    Relationship note linking to only 1 paper (which violates cross-feature synthesis).
    """
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    os.makedirs(wiki_dir, exist_ok=True)
    os.makedirs(synthesis_dir, exist_ok=True)
    os.makedirs(concepts_dir, exist_ok=True)
    
    # Pre-populate two valid Fact Notes
    fact1_content = """---
tags:
  - fact_note
---
# Fact Note: Paper A
"""
    fact2_content = """---
tags:
  - fact_note
---
# Fact Note: Paper B
"""
    with open(wiki_dir / 'Fact - Paper A.md', "w", encoding="utf-8") as f:
        f.write(fact1_content)
    with open(wiki_dir / 'Fact - Paper B.md', "w", encoding="utf-8") as f:
        f.write(fact2_content)
        
    mock_instance = MagicMock()
    mock_response = MagicMock()
    # The transcript only references ONE paper!
    mock_response.text = """# Synthesized Debate: Discussion
We only have one paper: [[Fact - Paper A]].
Meta-Analyst: This is a single paper analysis.
"""
    mock_instance.generate_content.return_value = mock_response
    mock_model_class.return_value = mock_instance
    
    # Run debate
    ret = run_debate_main([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir)
    ])
    
    # Verify that a 1-paper relationship was NOT silently generated,
    # or that the system correctly fell back or handled it.
    relationship_files = os.listdir(synthesis_dir)
    if relationship_files:
        relationship_note_path = synthesis_dir / relationship_files[0]
        with open(relationship_note_path, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.splitlines()
        closing_idx = -1
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                closing_idx = idx
                break
        yaml_block = "\n".join(lines[1:closing_idx])
        parsed = yaml.safe_load(yaml_block)
        
        # If it was generated, it MUST have both papers in source_papers (due to fallback)
        source_papers = parsed.get("source_papers", [])
        assert len(source_papers) >= 2, f"Relationship note should link to at least 2 papers, got: {source_papers}"
