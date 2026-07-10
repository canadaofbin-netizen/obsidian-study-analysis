import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# Ensure scripts directory is in python path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_scripts_dir = os.path.abspath(os.path.join(_current_dir, "..", "scripts"))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from extract_facts import save_state_incrementally
from run_debate import get_dynamic_personas
from compliance_checker import check_file

# =====================================================================
# Challenge 1: State Save PermissionError Retry & Fail Handling
# =====================================================================

@patch("time.sleep")
@patch("os.replace")
def test_save_state_incrementally_permission_error_eventual_success(mock_replace, mock_sleep, tmp_path):
    """
    Test that save_state_incrementally retries when PermissionError is raised
    and succeeds if the file lock is released before max_retries.
    """
    state_path = tmp_path / "processed_papers.json"
    
    # Simulate: 3 PermissionErrors, then a successful replace
    mock_replace.side_effect = [
        PermissionError("File locked by another process"),
        PermissionError("File locked by another process"),
        PermissionError("File locked by another process"),
        None
    ]
    
    result = save_state_incrementally(str(state_path), "paper_a.pdf")
    
    # It should have called replace 4 times, sleep 3 times, and returned the list
    assert mock_replace.call_count == 4
    assert mock_sleep.call_count == 3
    assert result == ["paper_a.pdf"]
    
    # Backoffs should be exponential: 0.1 * 2^0, 0.1 * 2^1, 0.1 * 2^2
    sleep_calls = [call_args[0][0] for call_args in mock_sleep.call_args_list]
    assert sleep_calls == [0.1, 0.2, 0.4]


@patch("time.sleep")
@patch("os.replace")
def test_save_state_incrementally_permission_error_exhausted(mock_replace, mock_sleep, tmp_path):
    """
    Test that save_state_incrementally retries up to 5 times when PermissionError
    is continuously raised, and returns None after exhausting retries.
    """
    state_path = tmp_path / "processed_papers.json"
    
    # Simulate: Continuous PermissionErrors
    mock_replace.side_effect = PermissionError("Permission denied")
    
    result = save_state_incrementally(str(state_path), "paper_b.pdf")
    
    # It should have retried 5 times and then returned None (catching the error)
    assert mock_replace.call_count == 5
    assert mock_sleep.call_count == 4  # No sleep after the final failure
    assert result is None
    
    # Backoffs should be exponential: 0.1, 0.2, 0.4, 0.8
    sleep_calls = [call_args[0][0] for call_args in mock_sleep.call_args_list]
    assert sleep_calls == [0.1, 0.2, 0.4, 0.8]


# =====================================================================
# Challenge 2: Specialized Persona Keyword Matching Gap
# =====================================================================

def test_specialized_persona_robotics_regex_gap(tmp_path):
    """
    Test that a fact note containing the word "robotics" successfully matches the
    regex in run_debate.py, causing the Robotics persona to be activated.
    """
    fact_note_path = tmp_path / "Fact - Robotics Research.md"
    # Content mentions "robotics" but not the word "robot" or "robots"
    fact_content = """---
tags:
  - fact_note
---
# Fact Note: Robotics Research
We study the application of robotics to industrial automation.
"""
    with open(fact_note_path, "w", encoding="utf-8") as f:
        f.write(fact_content)
        
    mock_logger = MagicMock()
    activated_personas = get_dynamic_personas([str(fact_note_path)], mock_logger)
    
    # This assertion verifies that "Robotics & Control Engineer" is activated,
    # confirming the regex gap is fixed.
    assert "Robotics & Control Engineer" in activated_personas


# =====================================================================
# Challenge 3: Compliance Bypass via Unclosed Code Blocks
# =====================================================================

def test_compliance_unclosed_code_block_bypass(tmp_path):
    """
    Test that an unclosed code block is detected and reported as a compliance violation.
    """
    bypass_file = tmp_path / "BypassNote.md"
    # Write a note with valid YAML frontmatter, but an unclosed code block
    # and subsequent violations (like header missing spaces and nested wikilinks)
    bad_content = """---
tags:
  - fact_note
---
# Header
```python
def foo():
    pass

#NoSpaceHeading
[[Fact - Valid [[NestedLink]]]]
"""
    with open(bypass_file, "w", encoding="utf-8") as f:
        f.write(bad_content)
        
    violations = check_file(str(bypass_file))
    
    # Verify that the unclosed code block fence violation is detected
    assert "File contains an unclosed code block fence" in violations


def test_extract_arxiv_id_query_parameters():
    """
    Test that extract_arxiv_id strips query parameters and fragment identifiers from arXiv URLs,
    returning the correct identifier.
    """
    from download_papers import extract_arxiv_id
    
    url_with_query = "http://arxiv.org/abs/2109.00001v1?context=cs.AI"
    extracted = extract_arxiv_id(url_with_query)
    
    # The expected output is '2109.00001', confirming query parameters are stripped.
    assert extracted == "2109.00001"

