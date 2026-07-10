import os
import sys
import pytest
import logging
from unittest.mock import patch, MagicMock

# Ensure scripts directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from run_debate import main as run_debate, get_dynamic_personas

def test_dynamic_personas_plural_boundary(tmp_path):
    """Verify that plural forms of keywords fail to activate specialized personas."""
    logger = logging.getLogger("TestLogger")
    
    # 1. Test plural keyword "transformers"
    fact1 = tmp_path / "Fact - Transformers.md"
    with open(fact1, "w", encoding="utf-8") as f:
        f.write("# Fact Note: Transformers\nThis paper discusses transformers and self-attention.")
    
    personas = get_dynamic_personas([str(fact1)], logger)
    # This assertion will FAIL if the plural word boundary bug is present
    assert "Deep Learning Architect" in personas

def test_dynamic_personas_plural_neural_networks(tmp_path):
    """Verify that plural keyword 'neural networks' fails to activate specialized personas."""
    logger = logging.getLogger("TestLogger")
    
    # 2. Test plural keyword "neural networks"
    fact2 = tmp_path / "Fact - Neural Networks.md"
    with open(fact2, "w", encoding="utf-8") as f:
        f.write("# Fact Note: Neural Networks\nThis paper focuses on neural networks.")
        
    personas = get_dynamic_personas([str(fact2)], logger)
    # This assertion will FAIL if the plural word boundary bug is present
    assert "Deep Learning Architect" in personas

def test_dynamic_personas_plural_robots(tmp_path):
    """Verify that plural keyword 'robots' fails to activate specialized personas."""
    logger = logging.getLogger("TestLogger")
    
    # 3. Test plural keyword "robots"
    fact3 = tmp_path / "Fact - Robots.md"
    with open(fact3, "w", encoding="utf-8") as f:
        f.write("# Fact Note: Robots\nThis paper is about robots.")
        
    personas = get_dynamic_personas([str(fact3)], logger)
    # This assertion will FAIL if the plural word boundary bug is present
    assert "Robotics & Control Engineer" in personas


def test_pipeline_system_exit_bypass(tmp_path, caplog):
    """Verify that SystemExit raised by run_debate is not caught by run_pipeline."""
    import run_pipeline
    
    with patch("run_pipeline.download_arxiv_papers"), \
         patch("run_pipeline.run_extraction", return_value=0), \
         patch("run_pipeline.run_debate", side_effect=SystemExit(1)):
         
        with pytest.raises(SystemExit):
            run_pipeline.main([
                '--query', 'all:AI',
                '--max-results', '0',
                '--pdf-dir', str(tmp_path / "pdf"),
                '--wiki-dir', str(tmp_path / "wiki"),
                '--log-path', str(tmp_path / "pipeline.log")
            ])
            
        # If SystemExit was bypassed (uncaught), no "Colosseum Debate failed" error log was produced.
        # We assert that the exception WAS caught and logged, which will FAIL because it wasn't.
        log_records = [r.message for r in caplog.records]
        assert any("Colosseum Debate failed" in msg for msg in log_records)

@patch('google.generativeai.GenerativeModel')
def test_debate_pipe_wikilink_handling(mock_model_class, tmp_path):
    """Verify that run_debate fails to update Fact Notes when WikiLinks contain a pipe '|' character."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    os.makedirs(wiki_dir)
    
    # Create fact note
    fact_content = "---\ntags:\n  - fact_note\n---\n# Fact Note: Paper One\n"
    with open(wiki_dir / "Fact - Paper One.md", "w", encoding="utf-8") as f:
        f.write(fact_content)
    with open(wiki_dir / "Fact - Paper Two.md", "w", encoding="utf-8") as f:
        f.write(fact_content)
        
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance
    mock_response = MagicMock()
    mock_response.text = """# Synthesized Debate: Theme
The core connection is optimization.
[[Fact - Paper One|Paper One Display Text]]
"""
    mock_instance.generate_content.return_value = mock_response
    
    run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir)
    ])
    
    with open(wiki_dir / "Fact - Paper One.md", "r", encoding="utf-8") as f:
        content = f.read()
        
    # This assertion will FAIL because the file was not updated due to the pipe character in the link.
    assert "Concept - Optimization" in content

@patch('google.generativeai.GenerativeModel')
def test_debate_empty_theme_name_fallback(mock_model_class, tmp_path):
    """Verify that run_debate fails to sanitize or fallback to a default theme name when theme contains only special characters."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    os.makedirs(wiki_dir)
    
    with open(wiki_dir / "Fact - Paper One.md", "w", encoding="utf-8") as f:
        f.write("# Fact Note One")
    with open(wiki_dir / "Fact - Paper Two.md", "w", encoding="utf-8") as f:
        f.write("# Fact Note Two")
        
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance
    mock_response = MagicMock()
    mock_response.text = """# Synthesized Debate: @#$%&*
The core connection is optimization.
[[Fact - Paper One]]
"""
    mock_instance.generate_content.return_value = mock_response
    
    run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir)
    ])
    
    # We assert that the generated file does NOT have an empty theme name "Relationship - .md".
    # This assertion will FAIL because "Relationship - .md" is indeed created.
    assert not os.path.exists(synthesis_dir / "Relationship - .md")


def test_compliance_checker_code_block_bypass(tmp_path):
    """Verify that compliance checker ignores brackets inside code blocks."""
    from compliance_checker import check_file
    
    # Test triple backticks with mismatched/nested brackets
    content1 = """---
tags:
  - fact_note
---
# Heading
```python
a = [[1, 2]
b = [[1, [[2]]]
```
"""
    f1 = tmp_path / "test1.md"
    with open(f1, "w", encoding="utf-8") as f:
        f.write(content1)
    
    violations1 = check_file(str(f1))
    assert not violations1, f"Expected no violations, got: {violations1}"
    
    # Test triple tildes with mismatched/nested brackets
    content2 = """---
tags:
  - fact_note
---
# Heading
~~~python
a = [[1, 2]
b = [[1, [[2]]]
~~~
"""
    f2 = tmp_path / "test2.md"
    with open(f2, "w", encoding="utf-8") as f:
        f.write(content2)
        
    violations2 = check_file(str(f2))
    assert not violations2, f"Expected no violations, got: {violations2}"


def test_horizontal_rule_body_text_preservation():
    """Verify that a horizontal rule in the body is not mistaken for a closing frontmatter boundary."""
    from extract_facts import clean_and_format_markdown
    
    content = """---
tags: fact_note
# My Heading
Some text here.
---
## Key Findings
"""
    result = clean_and_format_markdown(content)
    
    # The heading and body text must be preserved.
    assert "# My Heading" in result
    assert "Some text here." in result
    # We should have a valid frontmatter prepended and closed
    assert result.startswith("---")
    parts = result.split("---")
    # There should be at least 4 components when splitting by '---' (frontmatter start, middle content, frontmatter end, rest)
    assert len(parts) >= 4


@patch('google.generativeai.GenerativeModel')
def test_concept_title_fallback_consistency(mock_model_class, tmp_path):
    """Verify that single-paper fallback concept title logic is consistent (uses words[:4] if available)."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    os.makedirs(wiki_dir)
    
    # Create a single fact note with a 2-word title (4 or fewer words)
    with open(wiki_dir / "Fact - Short Title.md", "w", encoding="utf-8") as f:
        f.write("# Short Title")
    with open(wiki_dir / "Fact - Dummy.md", "w", encoding="utf-8") as f:
        f.write("# Fact Note Dummy")
        
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance
    mock_response = MagicMock()
    mock_response.text = """# Synthesized Debate: Short Theme
The debate happened.
[[Fact - Short Title]]
"""
    mock_instance.generate_content.return_value = mock_response
    
    run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir)
    ])
    
    assert os.path.exists(concepts_dir / "Concept - Short Title.md")
    assert not os.path.exists(concepts_dir / "Concept - Sequential Decision-Making.md")

