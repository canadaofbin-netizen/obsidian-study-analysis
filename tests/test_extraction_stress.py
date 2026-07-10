import os
import sys
import json
import pytest
import pypdf
import yaml
from unittest.mock import patch, MagicMock

# Ensure scripts directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from extract_facts import extract_text_from_pdf, clean_and_format_markdown, main as run_extraction
import extract_facts

@pytest.fixture(autouse=True)
def mock_pdf_reader_globally():
    # Override conftest's global mock so we can test actual PDF reader behavior!
    yield

# ---------------------------------------------------------
# Part 1: PDF Read Guard Testing (Encrypted, Zero-page, Empty)
# ---------------------------------------------------------

def test_encrypted_pdf_empty_password(tmp_path):
    """Verify decryption of PDF encrypted with an empty password."""
    pdf_path = tmp_path / "encrypted_empty.pdf"
    
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("")
    with open(pdf_path, "wb") as f:
        writer.write(f)
        
    # Should decrypt without raising ValueError
    text = extract_text_from_pdf(str(pdf_path))
    assert text.strip() == ""

def test_encrypted_pdf_non_empty_password(tmp_path):
    """Verify decryption fails and raises ValueError for PDF encrypted with non-empty password."""
    pdf_path = tmp_path / "encrypted_secret.pdf"
    
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("secret_pass")
    with open(pdf_path, "wb") as f:
        writer.write(f)
        
    with pytest.raises(ValueError) as exc_info:
        extract_text_from_pdf(str(pdf_path))
    assert "decryption with empty password failed" in str(exc_info.value)

def test_empty_pdf_file(tmp_path):
    """Verify handling of completely empty (0 bytes) PDF file."""
    pdf_path = tmp_path / "empty_0bytes.pdf"
    pdf_path.write_bytes(b"")
    
    # Under standard execution, PdfReader(empty_path) raises an exception.
    # extract_text_from_pdf should propagate this.
    with pytest.raises(Exception):
        extract_text_from_pdf(str(pdf_path))

def test_zero_page_pdf(tmp_path):
    """Verify behaviour with a PDF structure containing zero pages."""
    pdf_path = tmp_path / "zero_page.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n%%EOF")
        
    with patch("pypdf.PdfReader") as MockReader:
        mock_instance = MagicMock()
        mock_instance.is_encrypted = False
        mock_instance.pages = []
        MockReader.return_value = mock_instance
        
        text = extract_text_from_pdf(str(pdf_path))
        assert text == ""

# ---------------------------------------------------------
# Part 2: Markdown Clean and Format Parsing Testing
# ---------------------------------------------------------

def test_clean_triple_backticks():
    """Verify clean_and_format_markdown strips global code block wrappers."""
    # Case A: with markdown identifier
    content_a = """```markdown
---
tags:
  - fact_note
---
# Test Heading
```"""
    cleaned_a = clean_and_format_markdown(content_a)
    assert not cleaned_a.startswith("```")
    assert not cleaned_a.endswith("```")
    assert "tags:" in cleaned_a
    assert "# Test Heading" in cleaned_a

    # Case B: without language identifier
    content_b = """```
---
tags:
  - fact_note
---
# Test Heading
```"""
    cleaned_b = clean_and_format_markdown(content_b)
    assert not cleaned_b.startswith("```")
    assert not cleaned_b.endswith("```")

def test_clean_missing_header_spaces():
    """Verify that headings missing a space after '#' are standardized."""
    content = """---
tags:
  - fact_note
---
#Header 1
##Header 2
###Header 3
# Already OK
#1. Numbered list?
"""
    cleaned = clean_and_format_markdown(content)
    assert "# Header 1" in cleaned
    assert "## Header 2" in cleaned
    assert "### Header 3" in cleaned
    assert "# Already OK" in cleaned
    assert "# 1. Numbered list?" in cleaned

def test_clean_missing_closing_yaml_boundary():
    """Verify that missing closing frontmatter boundary is handled.
    Note: we are testing both with and without a header to expose potential issues.
    """
    # Case 1: missing closing boundary, but has a heading
    content_with_header = """---
tags:
  - tag1
# My Heading
Content here.
"""
    cleaned_with_header = clean_and_format_markdown(content_with_header)
    assert cleaned_with_header.count("---") == 2
    assert "  - tag1" in cleaned_with_header
    assert "  - fact_note" in cleaned_with_header
    assert "# My Heading" in cleaned_with_header

    # Case 2: missing closing boundary, and no heading
    content_no_header = """---
tags:
  - tag1
Only text here.
"""
    cleaned_no_header = clean_and_format_markdown(content_no_header)
    assert cleaned_no_header.count("---") == 2
    assert "tags:" in cleaned_no_header
    assert "  - tag1" in cleaned_no_header
    assert "  - fact_note" in cleaned_no_header
    assert "Only text here." in cleaned_no_header

def test_yaml_fallback_parsing():
    """Verify yaml parser fallback when yaml loading fails."""
    # E.g. invalid yaml syntax in tags
    content = """---
tags: [tag1, tag2,
---
# Heading
"""
    cleaned = clean_and_format_markdown(content)
    assert "tag1" in cleaned
    assert "tag2" in cleaned
    assert "fact_note" in cleaned

# ---------------------------------------------------------
# Part 3: Incremental State Saving Testing
# ---------------------------------------------------------

def test_incremental_state_saving(tmp_path):
    """Verify that state is saved incrementally after each paper is processed."""
    pdf_dir = tmp_path / "raw_sources"
    os.makedirs(pdf_dir)
    
    # Create 3 dummy pdf files
    pdf_1 = pdf_dir / "paper_1.pdf"
    pdf_2 = pdf_dir / "paper_2.pdf"
    pdf_3 = pdf_dir / "paper_3.pdf"
    for path in [pdf_1, pdf_2, pdf_3]:
        with open(path, "wb") as f:
            f.write(b"%PDF-1.4\n%%EOF")
            
    output_dir = tmp_path / "wiki"
    state_path = pdf_dir / "processed_papers.json"
    
    import google.generativeai as genai
    
    call_count = 0
    def mock_generate(self, contents, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if "paper_3.pdf" in contents:
            raise RuntimeError("API failure on paper 3")
        
        mock_resp = MagicMock()
        mock_resp.text = "---\ntags:\n  - fact_note\n---\n# Fact Note\nSuccess"
        return mock_resp
        
    # We patch PdfReader to return mock pages with valid-length text so they pass the length guard naturally
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is a dummy extracted PDF text that is long enough to pass the length guard check (minimum 100 characters)."
    
    with patch("pypdf.PdfReader") as MockReader, patch.object(genai.GenerativeModel, 'generate_content', mock_generate):
        mock_instance = MagicMock()
        mock_instance.is_encrypted = False
        mock_instance.pages = [mock_page]
        MockReader.return_value = mock_instance
        
        # We run the extraction
        run_extraction([
            '--pdf-dir', str(pdf_dir),
            '--output-dir', str(output_dir),
            '--state-path', str(state_path)
        ])
        
    # Check that state file exists
    assert os.path.exists(state_path)
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
        
    # State should contain paper_1.pdf and paper_2.pdf, but not paper_3.pdf
    assert "paper_1.pdf" in state
    assert "paper_2.pdf" in state
    assert "paper_3.pdf" not in state

def test_corrupted_file_skipped(tmp_path):
    """Verify that a corrupted (empty) file is skipped and does not trigger LLM calls, regardless of name length."""
    long_filename = "a" * 100 + ".pdf"
    pdf_path = tmp_path / long_filename
    pdf_path.write_bytes(b"") # Write 0 bytes so file exists but is corrupted/empty
    
    short_filename = "short.pdf"
    short_pdf_path = tmp_path / short_filename
    short_pdf_path.write_bytes(b"") # Write 0 bytes so file exists but is corrupted/empty
    
    import google.generativeai as genai
    mock_model = MagicMock()
    
    with patch.object(genai, 'GenerativeModel', return_value=mock_model):
        # 1. Short filename should be skipped when reading fails
        run_extraction([
            '--pdf-dir', str(short_pdf_path),
            '--output-dir', str(tmp_path / "wiki"),
            '--state-path', str(tmp_path / "state.json")
        ])
        mock_model.generate_content.assert_not_called()
        
        # 2. Long filename should also be skipped when reading fails (no fallback, no bypass)
        run_extraction([
            '--pdf-dir', str(pdf_path),
            '--output-dir', str(tmp_path / "wiki"),
            '--state-path', str(tmp_path / "state.json")
        ])
        mock_model.generate_content.assert_not_called()

def test_clean_markdown_conversational_triple_backticks():
    """Verify that conversational prefixes and suffixes, along with triple backticks, are fully cleaned."""
    content = """Here is the extracted fact note as requested:
```markdown
---
tags:
  - fact_note
---
# Heading
```
Hope this helps!"""
    cleaned = clean_and_format_markdown(content)
    # Both starting and trailing conversational texts, along with all backticks, should be stripped.
    assert "```" not in cleaned
    assert "Hope this helps!" not in cleaned
    assert "Here is the extracted" not in cleaned
    assert "tags:" in cleaned
    assert "# Heading" in cleaned

def test_clean_markdown_space_separated_tags():
    """Verify standardizing of space/comma-separated tags string in frontmatter."""
    content = """---
tags: tag1, tag2 tag3
---
# Heading
"""
    cleaned = clean_and_format_markdown(content)
    assert "  - fact_note" in cleaned
    assert "  - tag1" in cleaned
    assert "  - tag2" in cleaned
    assert "  - tag3" in cleaned


def test_short_text_pdf_skipped(tmp_path):
    """Verify that a PDF containing less than 100 characters of text is skipped and does not trigger LLM calls."""
    pdf_path = tmp_path / "short_content.pdf"
    
    # Write a dummy PDF structure
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n%%EOF")
        
    import google.generativeai as genai
    mock_model = MagicMock()
    
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is short text." # 19 characters
    
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

