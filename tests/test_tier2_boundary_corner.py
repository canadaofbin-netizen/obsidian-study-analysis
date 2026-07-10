import os
import sys
import json
import pytest
import pypdf
import urllib.error
import urllib.request
from unittest.mock import patch, MagicMock

# Ensure scripts directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from download_papers import download_arxiv_papers, make_request_with_retry
from extract_facts import main as run_extraction, clean_and_format_markdown, extract_text_from_pdf
from run_debate import main as run_debate
from compliance_checker import check_file, check_vault, main as run_compliance
from run_pipeline import main as run_pipeline

# ---------------------------------------------------------
# Feature 1: Paper Ingestion Boundary Tests
# ---------------------------------------------------------

@patch("time.sleep", return_value=None)
@patch("urllib.request.urlopen")
def test_ingestion_transient_failure_recovery(mock_urlopen, mock_sleep):
    """F1.1: Verify transient 503 recovery with exponential backoff."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = b"<feed><entry><title>Recovered Paper</title><link title='pdf' href='http://export.arxiv.org/pdf/2401.00001' /></entry></feed>"
    
    # Fail twice, succeed on third attempt
    mock_urlopen.side_effect = [
        urllib.error.HTTPError("http://arxiv.org", 503, "Service Unavailable", None, None),
        urllib.error.HTTPError("http://arxiv.org", 503, "Service Unavailable", None, None),
        mock_resp
    ]
    
    data = make_request_with_retry("http://arxiv.org/api/query", is_download=False)
    assert b"Recovered Paper" in data
    assert mock_urlopen.call_count == 3
    # Sleeps: 3s backoff, 6s backoff, then 3s success delay
    assert mock_sleep.call_count == 3
    mock_sleep.assert_any_call(3)
    mock_sleep.assert_any_call(6)

@patch("time.sleep", return_value=None)
@patch("urllib.request.urlopen")
def test_ingestion_persistent_failure_reraise(mock_urlopen, mock_sleep):
    """F1.2: Verify persistent network failures propagate after 3 attempts."""
    mock_urlopen.side_effect = urllib.error.HTTPError("http://arxiv.org", 503, "Service Unavailable", None, None)
    
    with pytest.raises(urllib.error.HTTPError):
        make_request_with_retry("http://arxiv.org/api/query", is_download=False)
        
    assert mock_urlopen.call_count == 3
    # Backoffs for attempt 1 (3s) and attempt 2 (6s) called
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(3)
    mock_sleep.assert_any_call(6)

@patch("download_papers.make_request_with_retry")
def test_ingestion_malformed_xml_feed(mock_make_request, tmp_path):
    """F1.3: Verify malformed XML response raises RuntimeError."""
    mock_make_request.return_value = b"<feed><entry><title>Malformed XML"
    
    with pytest.raises(RuntimeError) as excinfo:
        download_arxiv_papers(
            query="all:AI",
            max_results=1,
            output_dir=str(tmp_path / "out"),
            registry_path=str(tmp_path / "reg.json")
        )
    assert "Critical error fetching metadata" in str(excinfo.value)

def test_ingestion_corrupted_registry_recreation(tmp_path):
    """F1.4: Verify corrupted registry handles gracefully and is overwritten."""
    reg_path = tmp_path / "corrupted_registry.json"
    with open(reg_path, 'w', encoding='utf-8') as f:
        f.write("{invalid json: [")
        
    from download_papers import load_registry, save_registry
    loaded = load_registry(str(reg_path))
    assert loaded == {} # Starts fresh
    
    save_registry(str(reg_path), {"test_id": {"title": "Test"}})
    assert os.path.exists(reg_path)
    with open(reg_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert "test_id" in data

@patch("time.sleep", return_value=None)
@patch("urllib.request.urlretrieve")
def test_ingestion_filesystem_permission_propagation(mock_urlretrieve, mock_sleep, tmp_path):
    """F1.5: Verify local disk PermissionError propagates immediately without retry."""
    mock_urlretrieve.side_effect = PermissionError("Permission denied on disk")
    filepath = tmp_path / "denied.pdf"
    
    with pytest.raises(PermissionError):
        make_request_with_retry("http://arxiv.org/pdf/2401.00001.pdf", is_download=True, filepath=str(filepath))
        
    assert mock_urlretrieve.call_count == 1
    mock_sleep.assert_not_called()

# ---------------------------------------------------------
# Feature 2: Fact Extraction Boundary Tests
# ---------------------------------------------------------

def test_extraction_encrypted_pdf_handling(tmp_path):
    """F2.1: Verify encrypted PDF raises ValueError and skips."""
    pdf_path = tmp_path / "encrypted.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret")
    with open(pdf_path, "wb") as f:
        writer.write(f)
        
    from pypdf._reader import PdfReader as RealPdfReader
    with patch('pypdf.PdfReader', side_effect=RealPdfReader):
        with pytest.raises(ValueError) as excinfo:
            extract_text_from_pdf(str(pdf_path))
    assert "decryption with empty password failed" in str(excinfo.value)

def test_extraction_scanned_pdf_length_guard(tmp_path):
    """F2.2: Verify scanned/short PDF content (< 100 chars) is skipped."""
    pdf_path = tmp_path / "short.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n%%EOF")
        
    import google.generativeai as genai
    mock_model = MagicMock()
    
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Short content" # < 100 chars
    
    with patch("pypdf.PdfReader") as MockReader, patch.object(genai, 'GenerativeModel', return_value=mock_model):
        mock_instance = MagicMock()
        mock_instance.is_encrypted = False
        mock_instance.pages = [mock_page]
        MockReader.return_value = mock_instance
        
        run_extraction([
            '--pdf-dir', str(pdf_path),
            '--output-dir', str(tmp_path / "wiki"),
            '--state-path', str(tmp_path / "state.json")
        ])
        mock_model.generate_content.assert_not_called()

def test_extraction_gemini_api_failure_handling(tmp_path):
    """F2.3: Verify script resilience when Gemini API throws exception."""
    pdf_path = tmp_path / "valid.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n%%EOF")
        
    import google.generativeai as genai
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is a valid research paper text containing hypothesis and findings that exceeds one hundred characters limit."
    
    # Mock Gemini to raise Quota / Connection exception
    def mock_generate(*args, **kwargs):
        raise RuntimeError("Gemini API Quota Exceeded")
        
    with patch("pypdf.PdfReader") as MockReader, patch.object(genai.GenerativeModel, 'generate_content', mock_generate):
        mock_instance = MagicMock()
        mock_instance.is_encrypted = False
        mock_instance.pages = [mock_page]
        MockReader.return_value = mock_instance
        
        ret = run_extraction([
            '--pdf-dir', str(pdf_path),
            '--output-dir', str(tmp_path / "wiki"),
            '--state-path', str(tmp_path / "state.json")
        ])
        # Returns 0 (gracefully completed for the batch even if single paper failed)
        assert ret == 0
        # State should not mark the paper as successfully processed
        state_file = tmp_path / "state.json"
        assert not os.path.exists(state_file)

def test_extraction_malformed_llm_markdown_parsing():
    """F2.4: Verify markdown cleanup resolves conversational prefixes and bad tags."""
    malformed_output = """Here is the note:
```markdown
---
tags: tag1, tag2
---
#NoSpaceHeader
Content.
```"""
    cleaned = clean_and_format_markdown(malformed_output)
    assert cleaned.startswith("---")
    assert "  - tag1" in cleaned
    assert "  - tag2" in cleaned
    assert "  - fact_note" in cleaned
    assert "# NoSpaceHeader" in cleaned
    assert "Here is the note:" not in cleaned
    assert "```" not in cleaned

def test_extraction_state_merge_concurrency(tmp_path):
    """F2.5: Verify state is merged concurrently during incremental save."""
    state_path = tmp_path / "state.json"
    
    # Pre-populate state
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(["paper_existing.pdf"], f)
        
    from extract_facts import save_state_incrementally
    save_state_incrementally(str(state_path), "paper_new.pdf")
    
    with open(state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)
        
    assert "paper_existing.pdf" in state
    assert "paper_new.pdf" in state

# ---------------------------------------------------------
# Feature 3: Colosseum Debate Boundary Tests
# ---------------------------------------------------------

def test_debate_zero_fact_notes(tmp_path):
    """F3.1: Verify run_debate handles empty wiki directory gracefully by returning 0."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = tmp_path / "synthesis"
    concepts_dir = tmp_path / "concepts"
    os.makedirs(wiki_dir)
    
    ret = run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir)
    ])
    assert ret == 0

def test_debate_parsing_fallback(tmp_path):
    """F3.2: Verify fallback to defaults when Gemini outputs unrecognizable text."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = tmp_path / "synthesis"
    concepts_dir = tmp_path / "concepts"
    os.makedirs(wiki_dir)
    
    # Create two Fact Notes to avoid empty directory logic
    with open(wiki_dir / "Fact - Test.md", 'w', encoding='utf-8') as f:
        f.write("# Fact Note Test")
    with open(wiki_dir / "Fact - Test..md", 'w', encoding='utf-8') as f:
        f.write("# Fact Note Dummy")
        
    import google.generativeai as genai
    mock_resp = MagicMock()
    mock_resp.text = "This is a completely random response with no structure whatsoever."
    
    with patch.object(genai.GenerativeModel, 'generate_content', return_value=mock_resp):
        run_debate([
            '--wiki-dir', str(wiki_dir),
            '--synthesis-dir', str(synthesis_dir),
            '--concepts-dir', str(concepts_dir)
        ])
    # Defaults should be triggered
    assert os.path.exists(synthesis_dir / "Relationship - Test.md")
    assert os.path.exists(concepts_dir / "Concept - Test.md")

def test_debate_concept_link_deduplication(tmp_path):
    """F3.3: Verify that duplicate wikilinks are deduplicated in Concept Hub."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = tmp_path / "synthesis"
    concepts_dir = tmp_path / "concepts"
    os.makedirs(wiki_dir)
    
    # Create two Fact Notes to avoid early abort
    with open(wiki_dir / "Fact - PaperA.md", 'w', encoding='utf-8') as f:
        f.write("# Fact Note A")
    with open(wiki_dir / "Fact - Dummy.md", 'w', encoding='utf-8') as f:
        f.write("# Fact Note Dummy")
    
    import google.generativeai as genai
    mock_resp = MagicMock()
    mock_resp.text = """# Synthesized Debate: Agents
The core connection is the application of coordination.
References: [[Fact - PaperA]], [[Fact - PaperA]], [[Fact - PaperA]]"""
    
    with patch.object(genai.GenerativeModel, 'generate_content', return_value=mock_resp):
        run_debate([
            '--wiki-dir', str(wiki_dir),
            '--synthesis-dir', str(synthesis_dir),
            '--concepts-dir', str(concepts_dir)
        ])
        
    concept_hub = concepts_dir / "Concept - Coordination.md"
    assert os.path.exists(concept_hub)
    with open(concept_hub, 'r', encoding='utf-8') as f:
        content = f.read()
    # Should only contain the link once
    assert content.count("[[Fact - PaperA]]") == 1

def test_debate_unwritable_fact_note_warning(tmp_path):
    """F3.4: Verify debate execution proceeds if a Fact Note file is locked/deleted."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = tmp_path / "synthesis"
    concepts_dir = tmp_path / "concepts"
    os.makedirs(wiki_dir)
    
    # Create two Fact Notes
    with open(wiki_dir / "Fact - PaperA.md", 'w', encoding='utf-8') as f:
        f.write("# Fact Note A")
    with open(wiki_dir / "Fact - PaperB.md", 'w', encoding='utf-8') as f:
        f.write("# Fact Note B")
        
    import google.generativeai as genai
    mock_resp = MagicMock()
    mock_resp.text = """# Synthesized Debate: Theme
The core connection is optimization.
References: [[Fact - PaperA]], [[Fact - PaperB]]"""
    
    # Make writing to PaperA raise PermissionError
    original_open = open
    def mock_open_file(file, mode='r', *args, **kwargs):
        if "Fact - PaperA.md" in str(file) and 'w' in mode:
            raise PermissionError("Locked file")
        return original_open(file, mode, *args, **kwargs)
        
    with patch.object(genai.GenerativeModel, 'generate_content', return_value=mock_resp), \
         patch("builtins.open", mock_open_file):
        # Should not crash, exits 0
        run_debate([
            '--wiki-dir', str(wiki_dir),
            '--synthesis-dir', str(synthesis_dir),
            '--concepts-dir', str(concepts_dir)
        ])
    # PaperB should still have been updated
    with open(wiki_dir / "Fact - PaperB.md", 'r', encoding='utf-8') as f:
        content_b = f.read()
    assert "[[Concept - Optimization]]" in content_b

def test_debate_concept_filename_sanitization(tmp_path):
    """F3.5: Verify concept title containing invalid path characters is sanitized."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = tmp_path / "synthesis"
    concepts_dir = tmp_path / "concepts"
    os.makedirs(wiki_dir)
    
    # Create two Fact Notes to avoid early abort
    with open(wiki_dir / "Fact - PaperA.md", 'w', encoding='utf-8') as f:
        f.write("# Fact Note A")
    with open(wiki_dir / "Fact - Dummy.md", 'w', encoding='utf-8') as f:
        f.write("# Fact Note Dummy")
        
    import google.generativeai as genai
    mock_resp = MagicMock()
    # Title contains colon, question mark and slash
    mock_resp.text = """# Synthesized Debate: Sanity
The core connection is dynamic/adaptive: agent?control.
References: [[Fact - PaperA]]"""
    
    with patch.object(genai.GenerativeModel, 'generate_content', return_value=mock_resp):
        run_debate([
            '--wiki-dir', str(wiki_dir),
            '--synthesis-dir', str(synthesis_dir),
            '--concepts-dir', str(concepts_dir)
        ])
    # Sanitized filename should exclude colon, question mark, slash
    assert os.path.exists(concepts_dir / "Concept - Dynamicadaptive Agentcontrol.md")

# ---------------------------------------------------------
# Feature 4: Obsidian Compliance Boundary Tests
# ---------------------------------------------------------

def test_compliance_extra_frontmatter_delimiters(tmp_path):
    """F4.1: Verify compliance checker isolates the actual frontmatter from Horizontal Rules."""
    file_path = tmp_path / "horizontal_rule.md"
    content = """---
tags:
  - fact_note
---
# Title

This is some text.
---
This is a horizontal rule.
---
"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    violations = check_file(str(file_path))
    assert violations == []

def test_compliance_malformed_tags_format(tmp_path):
    """F4.2: Verify compliance checker catches malformed tag formatting."""
    # Case A: Tags as a string
    file_a = tmp_path / "string_tags.md"
    with open(file_a, 'w', encoding='utf-8') as f:
        f.write("---\ntags: single_tag\n---\n# Title\n")
    assert any("not a list" in v for v in check_file(str(file_a)))
    
    # Case B: Tags list block missing dashes but using flow list on new line
    file_b = tmp_path / "block_tags_no_dash.md"
    with open(file_b, 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  [tag1, tag2]\n---\n# Title\n")
    assert any("do not start with '-'" in v for v in check_file(str(file_b)))

def test_compliance_mismatched_nested_brackets(tmp_path):
    """F4.3: Verify detection of mismatched and nested WikiLinks."""
    # Nested WikiLinks
    nested_file = tmp_path / "nested.md"
    with open(nested_file, 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - fact_note\n---\n[[Fact - A [[Fact - B]]]]\n")
    violations = check_file(str(nested_file))
    assert any("Nested wikilinks" in v for v in violations)

def test_compliance_code_block_header_bypass(tmp_path):
    """F4.4: Verify headers check is bypassed inside code blocks."""
    code_block_file = tmp_path / "code_block.md"
    content = """---
tags:
  - fact_note
---
# Valid Title

```python
#NoSpaceComment in code block is fine
print("Hello")
```
"""
    with open(code_block_file, 'w', encoding='utf-8') as f:
        f.write(content)
    assert check_file(str(code_block_file)) == []

def test_compliance_comment_stripping_bypass(tmp_path):
    """F4.5: Verify bad headings/brackets are ignored inside comments."""
    commented_file = tmp_path / "comments.md"
    content = """---
tags:
  - fact_note
---
# Valid Title

<!-- #BadHeadingInsideComment -->
%% [[UnmatchedBracketInsideObsidianComment %%
"""
    with open(commented_file, 'w', encoding='utf-8') as f:
        f.write(content)
    assert check_file(str(commented_file)) == []

def test_compliance_mismatched_brackets_interleaved(tmp_path):
    """Verify out-of-order mismatched brackets are caught."""
    file_path = tmp_path / "interleaved.md"
    content = "---\ntags:\n  - fact_note\n---\n]] [[\n"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    violations = check_file(str(file_path))
    assert any("Mismatched" in v for v in violations)

def test_compliance_code_block_brackets(tmp_path):
    """Verify that brackets inside code blocks are ignored."""
    file_path = tmp_path / "code_block_brackets.md"
    content = """---
tags:
  - fact_note
---
```python
grid = [[1], [2]]
```
~~~
nested = [[3], [4]]
~~~
"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    violations = check_file(str(file_path))
    assert violations == []

def test_compliance_inline_code_brackets(tmp_path):
    """Verify that brackets inside inline code are ignored."""
    file_path = tmp_path / "inline_code_brackets.md"
    content = "---\ntags:\n  - fact_note\n---\nHere is inline code: `[[not_a_link]]` and ``[[another_one]]``.\n"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    violations = check_file(str(file_path))
    assert violations == []

def test_compliance_multiline_wikilink(tmp_path):
    """Verify cross-line wikilinks are flagged."""
    file_path = tmp_path / "multiline_wikilink.md"
    content = "---\ntags:\n  - fact_note\n---\n[[link\nwith newline]]\n"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    violations = check_file(str(file_path))
    assert any("cross-line" in v.lower() or "mismatched" in v.lower() for v in violations)

def test_compliance_heading_with_leading_spaces(tmp_path):
    """Verify headings with leading spaces are validated and flagged if missing space."""
    file_path = tmp_path / "indented_heading.md"
    content = "---\ntags:\n  - fact_note\n---\n  #NoSpaceHeading\n"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    violations = check_file(str(file_path))
    assert any("Heading format is incorrect" in v for v in violations)

def test_compliance_heading_in_blockquote(tmp_path):
    """Verify headings inside blockquotes are validated and flagged if missing space."""
    file_path = tmp_path / "blockquote_heading.md"
    content = "---\ntags:\n  - fact_note\n---\n> #NoSpaceHeading\n"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    violations = check_file(str(file_path))
    assert any("Heading format is incorrect" in v for v in violations)

def test_compliance_utf8_bom(tmp_path):
    """Verify UTF-8 BOM is stripped and does not cause compliance errors."""
    file_path = tmp_path / "bom.md"
    content = "\ufeff---\ntags:\n  - fact_note\n---\n# Valid Heading\n"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    violations = check_file(str(file_path))
    assert violations == []


# ---------------------------------------------------------
# Feature 5: E2E Integration Boundary Tests
# ---------------------------------------------------------

def test_pipeline_unwritable_log_dir(tmp_path):
    """F5.1: Verify run_pipeline fails gracefully when log directory is unwritable."""
    # Use an invalid path that cannot be created
    invalid_log = "/non_existent_folder_xyz/pipeline.log"
    
    with pytest.raises(Exception):
        run_pipeline([
            '--query', 'all:AI',
            '--max-results', '1',
            '--log-path', invalid_log
        ])

def test_pipeline_exits_on_compliance_failure(tmp_path):
    """F5.2: Verify pipeline runner exits with status 1 on compliance violations."""
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    log_path = tmp_path / "pipeline.log"
    
    # Mock compliance validation to return violations
    with patch("run_pipeline.check_vault", return_value={"Fact - Bad.md": ["Heading format incorrect"]}):
        with pytest.raises(SystemExit) as excinfo:
            run_pipeline([
                '--query', 'all:AI',
                '--max-results', '1',
                '--pdf-dir', str(pdf_dir),
                '--wiki-dir', str(wiki_dir),
                '--log-path', str(log_path)
            ])
        assert excinfo.value.code == 1

def test_pipeline_zero_papers_flow(tmp_path):
    """F5.3: Verify that 0 new papers ingestion flows through other steps safely."""
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    log_path = tmp_path / "pipeline.log"
    
    # Mock download_arxiv_papers to return empty dict
    with patch("run_pipeline.download_arxiv_papers", return_value={}), \
         patch("run_pipeline.run_extraction", return_value=0), \
         patch("run_pipeline.run_debate") as mock_debate:
         
        run_pipeline([
            '--query', 'all:AI',
            '--max-results', '0',
            '--pdf-dir', str(pdf_dir),
            '--wiki-dir', str(wiki_dir),
            '--log-path', str(log_path)
        ])
        
        # Debate is still executed (handles existing files or executes on empty list)
        assert mock_debate.called

def test_pipeline_partial_failure_resume(tmp_path):
    """F5.4: Verify pipeline avoids re-running ingestion/extraction on successful resume."""
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    log_path = tmp_path / "pipeline.log"
    state_path = pdf_dir / "processed_papers.json"
    registry_path = pdf_dir / "download_registry.json"
    
    os.makedirs(pdf_dir)
    # Simulate first run completed Ingestion and Extraction
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump({"2401.00001": {"title": "PaperA", "filename": "PaperA.pdf"}}, f)
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(["PaperA.pdf"], f)
        
    with patch("run_pipeline.download_arxiv_papers") as mock_ingest, \
         patch("run_pipeline.run_extraction", return_value=0) as mock_extract:
         
        run_pipeline([
            '--query', 'all:AI',
            '--max-results', '1',
            '--pdf-dir', str(pdf_dir),
            '--wiki-dir', str(wiki_dir),
            '--state-path', str(state_path),
            '--registry-path', str(registry_path),
            '--log-path', str(log_path)
        ])
        
        # Ingest should skip downloading because of registry match (download_arxiv_papers is still called,
        # but inside it skips urlretrieve. We verify download_arxiv_papers is called with correct arguments)
        assert mock_ingest.called
        # Extraction is called, but inside it will skip PaperA.pdf because of state_path check
        assert mock_extract.called

def test_pipeline_cli_parameter_boundaries(tmp_path):
    """F5.5: Verify pipeline behaves correctly with zero/negative results limits."""
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    log_path = tmp_path / "pipeline.log"
    
    with patch("run_pipeline.download_arxiv_papers") as mock_ingest:
        run_pipeline([
            '--query', 'all:AI',
            '--max-results', '0',
            '--pdf-dir', str(pdf_dir),
            '--wiki-dir', str(wiki_dir),
            '--log-path', str(log_path)
        ])
        # max_results = 0 is passed down
        mock_ingest.assert_called_once_with(
            query='all:AI',
            max_results=0,
            output_dir=str(pdf_dir),
            registry_path=os.path.join(str(pdf_dir), "download_registry.json")
        )
