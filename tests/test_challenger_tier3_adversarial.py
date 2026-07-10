import pytest
import sys
import os
import yaml
from unittest.mock import patch, MagicMock
from extract_facts import clean_and_format_markdown
from run_debate import main as run_debate_main

def test_adversarial_blockquote_heading():
    """
    Adversarial test: Verify if clean_and_format_markdown fails to correct
    headings when prefixed by markdown blockquotes (e.g. '>#Heading').
    """
    input_text = """---
tags:
  - fact_note
---
>#Invalid Blockquote Heading
"""
    output = clean_and_format_markdown(input_text)
    # The current implementation fails to insert a space, leaving it as '>#Invalid Blockquote Heading'
    assert "># Invalid Blockquote Heading" in output

def test_adversarial_multiple_code_blocks():
    """
    Adversarial test: Verify if clean_and_format_markdown gets hijacked
    by the first code block (e.g. an inline Python block) when the LLM
    returns multiple code blocks, leading to discarding the actual note.
    """
    input_text = """Some intro.
```python
# Py comment
def foo():
    pass
```
Here is the note:
```markdown
---
tags:
  - fact_note
---
# Fact Note: Real Title
```
"""
    output = clean_and_format_markdown(input_text)
    # The current implementation extracts the Python block and discards the markdown note
    assert "Fact Note: Real Title" in output

def test_adversarial_concept_link_in_code_block(tmp_path):
    """
    Adversarial test: Verify if run_debate incorrectly injects WikiLinks
    inside code blocks if the substring '## Related Concepts' matches there.
    """
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    os.makedirs(wiki_dir, exist_ok=True)
    os.makedirs(synthesis_dir, exist_ok=True)
    os.makedirs(concepts_dir, exist_ok=True)
    
    fact_content = """---
tags:
  - fact_note
---
# Fact Note: Test Paper 1

```python
# This is code
## Related Concepts
```
"""
    fact2_content = """---
tags:
  - fact_note
---
# Fact Note: Test Paper 2
"""
    with open(wiki_dir / "Fact - Test Paper 1.md", "w", encoding="utf-8") as f:
        f.write(fact_content)
    with open(wiki_dir / "Fact - Test Paper 2.md", "w", encoding="utf-8") as f:
        f.write(fact2_content)
        
    ret = run_debate_main([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir)
    ])
    assert ret == 0
    
    with open(wiki_dir / "Fact - Test Paper 1.md", "r", encoding="utf-8") as f:
        updated_content = f.read()
        
    lines = updated_content.splitlines()
    in_code = False
    corrupted = False
    for line in lines:
        if line.strip().startswith("```python"):
            in_code = True
        elif line.strip().startswith("```") and in_code:
            in_code = False
        elif in_code and "[[Concept -" in line:
            corrupted = True
            
    assert not corrupted, "Concept link was injected inside a code block!"
