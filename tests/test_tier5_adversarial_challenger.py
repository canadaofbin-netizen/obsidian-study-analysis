import os
import sys
import json
import pytest
import logging
from unittest.mock import patch, MagicMock

# Ensure scripts directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from extract_facts import save_state_incrementally
from run_debate import get_dynamic_personas
from compliance_checker import check_file

@patch('os.replace')
@patch('time.sleep')
def test_extract_facts_state_retry_success(mock_sleep, mock_replace, tmp_path):
    """Verify that save_state_incrementally retries and eventually succeeds on PermissionError."""
    state_path = str(tmp_path / "processed.json")
    
    # os.replace raises PermissionError twice, then succeeds (returns None)
    mock_replace.side_effect = [PermissionError("Locked"), PermissionError("Locked"), None]
    
    res = save_state_incrementally(state_path, "paper1.pdf")
    
    # It should have succeeded and returned the new state
    assert res == ["paper1.pdf"]
    assert mock_replace.call_count == 3
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(0.1)
    mock_sleep.assert_any_call(0.2)

@patch('os.replace')
@patch('time.sleep')
def test_extract_facts_state_retry_exhausted(mock_sleep, mock_replace, tmp_path):
    """Verify that save_state_incrementally retries up to max_retries on PermissionError and fails gracefully."""
    state_path = str(tmp_path / "processed.json")
    
    # os.replace raises PermissionError all 5 times
    mock_replace.side_effect = [PermissionError("Locked")] * 5
    
    res = save_state_incrementally(state_path, "paper1.pdf")
    
    # It should fail and return None
    assert res is None
    assert mock_replace.call_count == 5
    assert mock_sleep.call_count == 4  # called on attempts 0, 1, 2, 3
    mock_sleep.assert_any_call(0.1)
    mock_sleep.assert_any_call(0.2)
    mock_sleep.assert_any_call(0.4)
    mock_sleep.assert_any_call(0.8)

def test_run_debate_robotics_persona_gap(tmp_path):
    """Verify that specialized persona 'Robotics & Control Engineer' regex successfully matches the word 'robotics'."""
    logger = logging.getLogger("TestLogger")
    
    # Test case: "robotics" is present, activating the "Robotics & Control Engineer" persona
    fact_note_robotics = tmp_path / "Fact - Robotics Paper.md"
    with open(fact_note_robotics, "w", encoding="utf-8") as f:
        f.write("# Fact Note\nThis paper discusses robotics algorithms and execution pathways.")
        
    personas_robotics = get_dynamic_personas([str(fact_note_robotics)], logger)
    assert "Robotics & Control Engineer" in personas_robotics

    # Contrast case: "robot" is present, which also activates the persona
    fact_note_robot = tmp_path / "Fact - Robot Paper.md"
    with open(fact_note_robot, "w", encoding="utf-8") as f:
        f.write("# Fact Note\nThis paper discusses robot control theory.")
        
    personas_robot = get_dynamic_personas([str(fact_note_robot)], logger)
    assert "Robotics & Control Engineer" in personas_robot

def test_compliance_checker_unclosed_code_block_masking(tmp_path):
    """Verify that an unclosed code block fence is reported as a violation, while masking downstream formatting/link violations."""
    doc_path = tmp_path / "unclosed_block.md"
    
    # We construct a document with an unclosed code block fence.
    # All lines following the opening fence will be processed as part of the code block.
    # Thus, formatting and wikilink syntax violations on those lines are masked and ignored,
    # but the unclosed code block itself is detected.
    content = """---
tags:
  - fact_note
---
# Valid Title

```python
# Some valid python code

# The following lines violate compliance, but the code block is never closed:
#NoSpaceHeading
[[Fact - Missing Close
"""
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    violations = check_file(str(doc_path))
    # It should return exactly the unclosed code block fence violation
    assert violations == ["File contains an unclosed code block fence"], f"Expected unclosed code block violation, but got: {violations}"
