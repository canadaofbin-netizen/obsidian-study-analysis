import pytest
import sys
import os
import yaml
from unittest.mock import patch, MagicMock

# Ensure scripts directory is in python path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_scripts_dir = os.path.abspath(os.path.join(_current_dir, "..", "scripts"))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from extract_facts import clean_and_format_markdown
from run_debate import main as run_debate_main
from compliance_checker import check_vault

def test_clean_and_format_markdown_block_tags_no_spaces():
    """
    Adversarial test to verify if clean_and_format_markdown correctly handles
    YAML block format tags without leading spaces.
    """
    input_text = """---
tags:
- fact_note
- deep_learning
---
# Fact Note: Deep Learning for Autonomous Driving
"""
    output = clean_and_format_markdown(input_text)
    
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
    
    assert "-" not in parsed_yaml["tags"], f"Hyphen '-' should not be in tags: {parsed_yaml['tags']}"
    assert "fact_note" in parsed_yaml["tags"]
    assert "deep_learning" in parsed_yaml["tags"]

def test_clean_and_format_markdown_preceding_horizontal_rule():
    """
    Adversarial test to verify if clean_and_format_markdown gets confused by
    a horizontal rule preceding the main code block.
    """
    input_text = """Below are the extracted facts.
---
```yaml
---
tags:
  - fact_note
---
# Fact Note: Title
```"""
    output = clean_and_format_markdown(input_text)
    
    # The output should NOT contain backticks or "Below are the extracted facts"
    # and should be a clean, compliant markdown file.
    assert "Below are the extracted facts" not in output, "Introductory text should be stripped"
    assert "```" not in output, "Fenced code block wrappers should be stripped"

def test_debate_on_single_fact_note(tmp_path):
    """
    Adversarial test to verify if run_debate runs and generates a Concept Hub
    when only ONE Fact Note exists in the wiki directory, which violates Rule 2
    of SYSTEM_HARNESS.md.
    """
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    os.makedirs(wiki_dir, exist_ok=True)
    os.makedirs(synthesis_dir, exist_ok=True)
    os.makedirs(concepts_dir, exist_ok=True)
    
    # Pre-populate ONLY ONE Fact Note
    fact_content = """---
tags:
  - fact_note
---
# Fact Note: Single Paper Test
"""
    with open(wiki_dir / "Fact - Single Paper Test.md", "w", encoding="utf-8") as f:
        f.write(fact_content)
        
    ret = run_debate_main([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir)
    ])
    
    concept_files = os.listdir(concepts_dir)
    assert len(concept_files) == 0, f"No Concept Hub should be generated for a single paper, but got: {concept_files}"

@patch('google.generativeai.GenerativeModel')
def test_debate_with_double_quotes_in_title(mock_model_class, tmp_path):
    """
    Adversarial test to verify if run_debate correctly escapes double quotes
    in paper titles when generating the YAML frontmatter.
    """
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    os.makedirs(wiki_dir, exist_ok=True)
    os.makedirs(synthesis_dir, exist_ok=True)
    os.makedirs(concepts_dir, exist_ok=True)
    
    # Pre-populate two valid Fact Notes (with Windows-compliant filenames)
    fact1_content = """---
tags:
  - fact_note
---
# Fact Note: AI Agent Systems
"""
    fact2_content = """---
tags:
  - fact_note
---
# Fact Note: Other Study
"""
    with open(wiki_dir / 'Fact - AI Agent Systems.md', "w", encoding="utf-8") as f:
        f.write(fact1_content)
    with open(wiki_dir / 'Fact - Other Study.md', "w", encoding="utf-8") as f:
        f.write(fact2_content)
        
    # Setup mock LLM response containing double quotes in the extracted paper title
    mock_instance = MagicMock()
    mock_response = MagicMock()
    # The transcript references a paper title containing double quotes
    mock_response.text = """# Synthesized Debate: Discussion
We have papers: [[Fact - AI "Agent" Systems]], [[Fact - Other Study]].
Meta-Analyst: The intersection is Decision-making.
"""
    mock_instance.generate_content.return_value = mock_response
    mock_model_class.return_value = mock_instance
    
    ret = run_debate_main([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir)
    ])
    assert ret == 0
    
    relationship_files = os.listdir(synthesis_dir)
    assert len(relationship_files) >= 1
    relationship_note_path = synthesis_dir / relationship_files[0]
    
    with open(relationship_note_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Extract YAML frontmatter
    lines = content.splitlines()
    assert lines[0] == "---"
    closing_idx = -1
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_idx = idx
            break
    assert closing_idx != -1
    
    yaml_block = "\n".join(lines[1:closing_idx])
    try:
        parsed = yaml.safe_load(yaml_block)
        assert parsed is not None
    except Exception as e:
        pytest.fail(f"YAML frontmatter is malformed due to unescaped double quotes: {e}")
