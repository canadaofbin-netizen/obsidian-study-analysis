import os
import json
import pytest
import tempfile
import urllib.error
from unittest.mock import patch, MagicMock
from scripts.download_papers import (
    extract_arxiv_id,
    load_registry,
    save_registry,
    make_request_with_retry,
    download_arxiv_papers
)

def test_extract_arxiv_id():
    """Verify that extract_arxiv_id correctly parses and cleans arXiv identifiers."""
    assert extract_arxiv_id("http://arxiv.org/abs/2109.00001v1") == "2109.00001"
    assert extract_arxiv_id("http://arxiv.org/abs/math/0211111v12") == "math/0211111"
    assert extract_arxiv_id("http://arxiv.org/pdf/2109.00001") == "2109.00001"
    assert extract_arxiv_id("2109.00001") == "2109.00001"
    assert extract_arxiv_id(None) is None

def test_load_save_registry():
    """Verify that registry load and save operations function correctly and handle errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg_path = os.path.join(tmpdir, "registry.json")
        
        # Test loading missing registry
        assert load_registry(reg_path) == {}
        
        # Test saving registry
        test_data = {"1234.5678": {"title": "Test Title", "filename": "test.pdf"}}
        save_registry(reg_path, test_data)
        
        # Test loading existing registry
        loaded = load_registry(reg_path)
        assert loaded == test_data
        
        # Test loading invalid json registry
        with open(reg_path, 'w', encoding='utf-8') as f:
            f.write("invalid json")
        assert load_registry(reg_path) == {}

@patch("time.sleep", return_value=None)
@patch("urllib.request.urlopen")
def test_make_request_with_retry_query_success(mock_urlopen, mock_sleep):
    """Verify that querying with retry succeeds on the first attempt and enforces a delay."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"<feed>success</feed>"
    mock_urlopen.return_value.__enter__.return_value = mock_resp
    
    data = make_request_with_retry("http://test.url", is_download=False)
    assert data == b"<feed>success</feed>"
    mock_sleep.assert_called_once_with(3)

@patch("time.sleep", return_value=None)
@patch("urllib.request.urlopen")
def test_make_request_with_retry_query_exponential_backoff(mock_urlopen, mock_sleep):
    """Verify that querying retries on transient errors with exponential backoff."""
    # First attempt fails with 503, second succeeds
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = b"<feed>success</feed>"
    
    mock_urlopen.side_effect = [
        urllib.error.HTTPError("http://test.url", 503, "Service Unavailable", None, None),
        mock_resp
    ]
    
    data = make_request_with_retry("http://test.url", is_download=False)
    assert data == b"<feed>success</feed>"
    # Sleeps: 3s backoff, then 3s delay after success
    mock_sleep.assert_any_call(3)
    assert mock_sleep.call_count == 2

@patch("time.sleep", return_value=None)
@patch("urllib.request.urlopen")
def test_make_request_with_retry_query_failures_reraise(mock_urlopen, mock_sleep):
    """Verify that make_request_with_retry eventually reraises the exception if all attempts fail."""
    mock_urlopen.side_effect = urllib.error.HTTPError("http://test.url", 503, "Service Unavailable", None, None)
    
    with pytest.raises(urllib.error.HTTPError):
        make_request_with_retry("http://test.url", is_download=False)
    
    # Retried 3 times, backoffs: 3s, 6s
    assert mock_urlopen.call_count == 3
    mock_sleep.assert_any_call(3)
    mock_sleep.assert_any_call(6)

@patch("time.sleep", return_value=None)
@patch("urllib.request.urlretrieve")
@patch("os.replace")
def test_make_request_with_retry_download_atomic(mock_replace, mock_urlretrieve, mock_sleep):
    """Verify that downloading uses a temporary file and atomically renames it upon success."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "final.pdf")
        make_request_with_retry("http://test.url/pdf", is_download=True, filepath=filepath)
        
        mock_urlretrieve.assert_called_once_with("http://test.url/pdf", filepath + ".tmp")
        mock_replace.assert_called_once_with(filepath + ".tmp", filepath)
        mock_sleep.assert_called_once_with(3)

@patch("scripts.download_papers.make_request_with_retry")
def test_download_arxiv_papers_flow(mock_make_request):
    """Verify the entire download flow: duplicates registry lookup, disk lookup, registration, and downloads."""
    mock_xml = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <title>New AI Agent Paper</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.00001" />
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2401.00002v1</id>
        <title>Existing Disk Paper</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.00002" />
      </entry>
      <entry>
        <id>http://arxiv.org/abs/2401.00003v1</id>
        <title>Registered Paper</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.00003" />
      </entry>
    </feed>
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "sources")
        registry_path = os.path.join(output_dir, "registry.json")
        
        # Pre-populate registry with paper 2401.00003
        os.makedirs(output_dir, exist_ok=True)
        initial_registry = {
            "2401.00003": {
                "title": "Registered Paper",
                "filename": "Registered Paper.pdf",
                "registered_at": "2026-07-10T00:00:00Z"
            }
        }
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(initial_registry, f)
            
        # Pre-create paper 2401.00002 on disk (but not in registry)
        disk_paper_path = os.path.join(output_dir, "Existing Disk Paper.pdf")
        with open(disk_paper_path, 'wb') as f:
            f.write(b"%PDF-1.4...")
            
        # Mock request return values
        mock_make_request.side_effect = [
            mock_xml.encode('utf-8'), # XML feed request
            True                      # PDF download request for 2401.00001
        ]
        
        # Run download logic
        download_arxiv_papers(
            query="all:AI",
            max_results=3,
            output_dir=output_dir,
            registry_path=registry_path
        )
        
        # Assertions
        # 1. 2401.00001 was downloaded
        mock_make_request.assert_any_call("http://export.arxiv.org/pdf/2401.00001.pdf", is_download=True, filepath=os.path.join(output_dir, "New AI Agent Paper.pdf"))
        
        # 2. 2401.00002 (on disk) was not downloaded but registered in JSON
        # 3. 2401.00003 (in registry) was not downloaded
        with open(registry_path, 'r', encoding='utf-8') as f:
            final_registry = json.load(f)
            
        assert "2401.00001" in final_registry
        assert "2401.00002" in final_registry
        assert "2401.00003" in final_registry
        assert final_registry["2401.00002"]["title"] == "Existing Disk Paper"


def test_clean_markdown_code_block_comments():
    """Verify that clean_and_format_markdown does not format comments inside code blocks."""
    from scripts.extract_facts import clean_and_format_markdown
    text = """---
tags:
  - fact_note
---
# Real Heading

```python
# This is a comment inside a code block
import os
```

~~~bash
#!/bin/bash
# Another comment
~~~
"""
    cleaned = clean_and_format_markdown(text)
    assert "# This is a comment inside a code block" in cleaned
    assert "# Another comment" in cleaned
    assert "# Real Heading" in cleaned


def test_compliance_checker_comments():
    """Verify compliance_checker strips HTML and Obsidian comments and avoids false positives."""
    from scripts.compliance_checker import check_file
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.md")
        content = """---
tags:
  - fact_note
---
# Valid Heading
<!-- #BadHeading -->
%% [[MismatchedBracket %%
%% [[]] %%
<!-- [[Nested [[Link]]]] -->
"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        violations = check_file(file_path)
        assert violations == [], f"Expected no violations, got: {violations}"


def test_compliance_checker_yaml_tags():
    """Verify compliance_checker YAML tag check robustness."""
    from scripts.compliance_checker import check_file
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Case A: Inline list formatting of tags (should pass)
        file_path_inline = os.path.join(tmpdir, "inline.md")
        content_inline = """---
tags: [tag1, tag2]
author: Jane Doe
---
# Valid Title
"""
        with open(file_path_inline, 'w', encoding='utf-8') as f:
            f.write(content_inline)
        assert check_file(file_path_inline) == []
        
        # Case B: Block format of tags (should pass)
        file_path_block = os.path.join(tmpdir, "block.md")
        content_block = """---
tags:
  - tag1
  - tag2
author: Jane Doe
---
# Valid Title
"""
        with open(file_path_block, 'w', encoding='utf-8') as f:
            f.write(content_block)
        assert check_file(file_path_block) == []

        # Case C: Invalid block format (fails tags on new lines without dash)
        file_path_invalid = os.path.join(tmpdir, "invalid.md")
        content_invalid = """---
tags:
  tag1
  tag2
---
# Valid Title
"""
        with open(file_path_invalid, 'w', encoding='utf-8') as f:
            f.write(content_invalid)
        violations = check_file(file_path_invalid)
        assert len(violations) > 0


def test_run_debate_concept_link_insertion_and_deduplication(tmp_path):
    """Verify that run_debate deduplicates links in concept hubs and inserts concept link directly under ## Related Concepts header."""
    from scripts.run_debate import main as run_debate
    import google.generativeai as genai
    from unittest.mock import patch, MagicMock
    
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = tmp_path / "synthesis"
    concepts_dir = tmp_path / "concepts"
    os.makedirs(wiki_dir)
    
    # Create Fact Notes, one with ## Related Concepts already present
    fact1_path = wiki_dir / "Fact - Paper One.md"
    with open(fact1_path, 'w', encoding='utf-8') as f:
        f.write("""---
tags:
  - fact_note
---
# Fact Note: Paper One

## Related Concepts
* [[Concept - Existing]]

Some other content at the end of the file.
""")
    with open(wiki_dir / "Fact - Paper Two.md", 'w', encoding='utf-8') as f:
        f.write("""---
tags:
  - fact_note
---
# Fact Note: Paper Two
""")
        
    # Mock Gemini response to cite Fact - Paper One multiple times
    mock_resp = MagicMock()
    mock_resp.text = """# Synthesized Debate: Dynamic Agent Allocation
The core connection is the application of sequential decision making.
Meta-Analyst: As discussed in [[Fact - Paper One]], agent allocation is key.
Devil's Advocate: But [[Fact - Paper One]] also points out limitations.
"""
    
    with patch.object(genai.GenerativeModel, 'generate_content', return_value=mock_resp):
        run_debate([
            '--wiki-dir', str(wiki_dir),
            '--synthesis-dir', str(synthesis_dir),
            '--concepts-dir', str(concepts_dir)
        ])
        
    # Verify concept hub file exists
    import glob
    concept_files = glob.glob(str(concepts_dir / "Concept - *.md"))
    assert len(concept_files) == 1
    concept_path = concept_files[0]
    
    # Verify deduplication of source papers in concept hub
    with open(concept_path, 'r', encoding='utf-8') as f:
        hub_content = f.read()
    assert hub_content.count("[[Fact - Paper One]]") == 1
    
    # Verify correct concept link placement under header (not at EOF)
    concept_title = os.path.basename(concept_path).replace("Concept - ", "").replace(".md", "")
    with open(fact1_path, 'r', encoding='utf-8') as f:
        fact_content = f.read()
        
    assert f"## Related Concepts\n* [[Concept - {concept_title}]]\n* [[Concept - Existing]]" in fact_content


@patch("time.sleep", return_value=None)
@patch("urllib.request.urlopen")
def test_make_request_with_retry_disk_errors_propagate_immediately(mock_urlopen, mock_sleep):
    """Verify that filesystem/disk errors propagate immediately without retrying."""
    # Raise PermissionError on first attempt
    mock_urlopen.side_effect = PermissionError("Permission denied")
    
    with pytest.raises(PermissionError):
        make_request_with_retry("http://test.url", is_download=False)
        
    # Should only attempt once and not sleep
    assert mock_urlopen.call_count == 1
    mock_sleep.assert_not_called()
