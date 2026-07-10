import os
import sys
import pytest
import urllib.error
from unittest.mock import patch, MagicMock

# Ensure scripts directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from extract_facts import clean_and_format_markdown
from download_papers import download_arxiv_papers, make_request_with_retry
from run_debate import main as run_debate

# ----------------------------------------------------------------------
# 1. Horizontal Rule Truncation in Fact Extraction
# ----------------------------------------------------------------------
def test_clean_markdown_horizontal_rule_truncation():
    """Verify that a horizontal rule in the body without code blocks does NOT truncate content."""
    content = """# Paper Title
This is the introduction.

---

## Key Findings
- Finding A
- Finding B
"""
    cleaned = clean_and_format_markdown(content)
    
    # Verify that the introduction and title are preserved
    assert "This is the introduction." in cleaned
    assert "# Paper Title" in cleaned
    assert "## Key Findings" in cleaned


# ----------------------------------------------------------------------
# 2. Empty Sanitized Theme and Concept File Naming & Fallback
# ----------------------------------------------------------------------
@patch('google.generativeai.GenerativeModel')
def test_run_debate_empty_sanitized_theme_and_concept(mock_model_class, tmp_path):
    """Verify that empty sanitized titles fall back to default theme and concept names."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = tmp_path / "synthesis"
    concepts_dir = tmp_path / "concepts"
    os.makedirs(wiki_dir)
    
    # Create valid fact note to trigger the debate run
    with open(wiki_dir / "Fact - Paper One.md", "w", encoding="utf-8") as f:
        f.write("# Fact Note: Paper One\n## Related Concepts\n")
    with open(wiki_dir / "Fact - Paper Two.md", "w", encoding="utf-8") as f:
        f.write("# Fact Note: Paper Two\n## Related Concepts\n")
        
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance
    mock_response = MagicMock()
    
    # Provide a transcript where theme and concept name are only special characters/symbols (e.g. !!!)
    mock_response.text = """# Synthesized Debate: !!!
The core connection is !!!.
[[Fact - Paper One]]
"""
    mock_instance.generate_content.return_value = mock_response
    
    # Run the debate script
    code = run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir),
    ])
    
    assert code == 0
    
    # Verify that no empty file names are created
    empty_rel_path = synthesis_dir / "Relationship - .md"
    empty_concept_path = concepts_dir / "Concept - .md"
    
    assert not os.path.exists(empty_rel_path)
    assert not os.path.exists(empty_concept_path)
    
    # Verify they fell back to the defaults
    default_rel_path = synthesis_dir / "Relationship - AI Agent Advancements.md"
    default_concept_path = concepts_dir / "Concept - Sequential Decision-Making.md"
    
    assert os.path.exists(default_rel_path)
    assert os.path.exists(default_concept_path)
    
    # Verify the updated Fact note contains links to the fallback defaults
    with open(wiki_dir / "Fact - Paper One.md", "r", encoding="utf-8") as f:
        fact_content = f.read()
        
    assert "[[Concept - Sequential Decision-Making]]" in fact_content
    assert "[[Relationship - AI Agent Advancements]]" in fact_content


# ----------------------------------------------------------------------
# 3. Urllib Temp File Leakage on Download Failure
# ----------------------------------------------------------------------
@patch("urllib.request.urlretrieve")
def test_temp_file_leak_on_failure(mock_urlretrieve, tmp_path):
    """Verify that a temporary PDF download file is cleaned up if urlretrieve fails."""
    filepath = tmp_path / "target_paper.pdf"
    tmp_filepath = tmp_path / "target_paper.pdf.tmp"
    
    # Simulate partial file creation during download
    def side_effect(url, filename):
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, "w") as f:
            f.write("partial downloaded data")
        raise urllib.error.URLError("Network timed out")
        
    mock_urlretrieve.side_effect = side_effect
    
    with pytest.raises(urllib.error.URLError):
        make_request_with_retry("http://arxiv.org/pdf/2401.00001.pdf", is_download=True, filepath=str(filepath))
        
    # Verify the temporary download file was cleaned up and deleted
    assert not os.path.exists(tmp_filepath)


# ----------------------------------------------------------------------
# 4. Inconsistent Fallback Concept Title Slicing Heuristic
# ----------------------------------------------------------------------
@patch('google.generativeai.GenerativeModel')
def test_brittle_concept_title_fallback(mock_model_class, tmp_path):
    """Verify how the fallback concept title behaves under 4-word vs 5-word titles using first N words."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = tmp_path / "synthesis"
    concepts_dir = tmp_path / "concepts"
    os.makedirs(wiki_dir)
    
    # 4-word title: Fact - One Two Three Four.md -> clean title "One Two Three Four"
    with open(wiki_dir / "Fact - One Two Three Four.md", "w", encoding="utf-8") as f:
        f.write("# Fact Note\n")
    with open(wiki_dir / "Fact - Dummy.md", "w", encoding="utf-8") as f:
        f.write("# Fact Note Dummy\n")
        
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance
    mock_response = MagicMock()
    # No matching core connection keywords
    mock_response.text = """# Synthesized Debate: Theme
No matching keyword connection here.
"""
    mock_instance.generate_content.return_value = mock_response
    
    run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir),
    ])
    
    # For a 4-word title, it should use the first 4 words: "One Two Three Four"
    assert os.path.exists(concepts_dir / "Concept - One Two Three Four.md")
    
    # Cleanup for next case
    os.remove(concepts_dir / "Concept - One Two Three Four.md")
    os.remove(synthesis_dir / "Relationship - Theme.md")
    os.remove(wiki_dir / "Fact - One Two Three Four.md")
    
    # 5-word title: Fact - One Two Three Four Five.md -> clean title "One Two Three Four Five"
    with open(wiki_dir / "Fact - One Two Three Four Five.md", "w", encoding="utf-8") as f:
        f.write("# Fact Note\n")
    with open(wiki_dir / "Fact - Dummy.md", "w", encoding="utf-8") as f:
        f.write("# Fact Note Dummy\n")
        
    run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir),
    ])
    
    # For a 5-word title, it should use the first 4 words: "One Two Three Four" instead of "Five"
    assert os.path.exists(concepts_dir / "Concept - One Two Three Four.md")
    assert not os.path.exists(concepts_dir / "Concept - Five.md")
