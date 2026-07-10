import os
import sys
import pytest
import random
import time

# Ensure scripts directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from compliance_checker import check_file

def test_compliance_extremely_large_document(tmp_path):
    """Verify compliance checker handles extremely large documents without CPU/memory exhaustion or crashes."""
    large_file = tmp_path / "large_doc.md"
    
    # Construct a large document with 50,000 lines
    lines = [
        "---",
        "tags:",
        "  - fact_note",
        "---",
        "# Large Document Title",
    ]
    # Add 50,000 lines of standard content, code blocks, comments, and links
    for i in range(50000):
        if i % 1000 == 0:
            lines.append(f"## Section {i}")
        elif i % 1000 == 1:
            lines.append(f"#NoSpaceHeading{i}") # Violation line
        elif i % 1000 == 2:
            lines.append("```python\n#NoSpaceComment in code block is fine\n```")
        elif i % 1000 == 3:
            lines.append("<!-- html comment here -->")
        elif i % 1000 == 4:
            lines.append(f"[[Fact - Note {i}]]")
        elif i % 1000 == 5:
            lines.append("[[Fact - Nested [[Fact - Inner]] ]]") # Violation line
        else:
            lines.append("Standard markdown text with some words.")
            
    with open(large_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        
    start_time = time.perf_counter()
    violations = check_file(str(large_file))
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    # Assert it processes a 50k line file quickly (e.g., within 2 seconds)
    assert duration < 2.0, f"Performance too slow: took {duration:.2f} seconds"
    assert len(violations) > 0
    # Check that it found the specific violations
    assert any("Heading format is incorrect" in v for v in violations)
    assert any("Nested wikilinks" in v for v in violations)

def test_compliance_highly_nested_markdown(tmp_path):
    """Verify checker handles deeply nested blockquotes/lists without recursion limit crashes."""
    nested_file = tmp_path / "nested_doc.md"
    
    # 5,000 levels of nested blockquotes
    nested_quotes = "> " * 5000 + "#HeadingWithoutSpace"
    
    # 5,000 levels of lists
    nested_lists = "  " * 5000 + "- #HeadingWithoutSpaceInList"
    
    content = f"""---
tags:
  - fact_note
---
# Title

{nested_quotes}

{nested_lists}
"""
    with open(nested_file, 'w', encoding='utf-8') as f:
        f.write(content)
        
    # Should not raise RecursionError
    violations = check_file(str(nested_file))
    assert len(violations) >= 2
    assert any("Heading format is incorrect" in v for v in violations)

def test_compliance_mismatched_wikilinks_combinatorics(tmp_path):
    """Verify checker handles pathological wikilink syntax without crashes."""
    file_path = tmp_path / "wikilinks_pathological.md"
    
    # Case 1: 5,000 open brackets followed by 5,000 close brackets
    # This checks for performance and stack correctness.
    pathological_brackets = "[[" * 5000 + "]]" * 5000
    
    # Case 2: Lots of nested and orphaned combinations
    nested_and_orphaned = "[[foo[[bar]] [[baz]] ]] [[nested[[inner]]"
    
    content = f"""---
tags:
  - fact_note
---
# Title

{pathological_brackets}

{nested_and_orphaned}
"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    violations = check_file(str(file_path))
    assert len(violations) > 0
    assert any("Nested wikilinks" in v for v in violations) or any("orphaned" in v for v in violations)

def test_compliance_malformed_yaml_frontmatter(tmp_path):
    """Verify checker is resilient to malformed/corrupted YAML frontmatters."""
    # Case A: Syntax error in YAML (tabs, invalid indentation)
    file_a = tmp_path / "yaml_syntax_error.md"
    content_a = """---
tags:
\t- tab_character_not_allowed_in_yaml
\t- another_tag
---
# Title
"""
    with open(file_a, 'w', encoding='utf-8') as f:
        f.write(content_a)
    violations_a = check_file(str(file_a))
    assert any("Failed to parse YAML frontmatter" in v for v in violations_a)

    # Case B: Non-mapping structure at top level
    file_b = tmp_path / "yaml_non_dict.md"
    content_b = """---
- list_item_1
- list_item_2
---
# Title
"""
    with open(file_b, 'w', encoding='utf-8') as f:
        f.write(content_b)
    violations_b = check_file(str(file_b))
    assert any("YAML frontmatter is not a dictionary/mapping" in v for v in violations_b)

    # Case C: Tags is a list containing non-string or empty elements
    file_c = tmp_path / "yaml_bad_tag_items.md"
    content_c = """---
tags:
  - ok_tag
  - 123
  - ""
---
# Title
"""
    with open(file_c, 'w', encoding='utf-8') as f:
        f.write(content_c)
    violations_c = check_file(str(file_c))
    assert any("contains invalid" in v for v in violations_c)

    # Case D: Empty tags list (should be fine if empty list, but invalid if not a list)
    file_d = tmp_path / "yaml_empty_tags.md"
    content_d = """---
tags: []
---
# Title
"""
    with open(file_d, 'w', encoding='utf-8') as f:
        f.write(content_d)
    violations_d = check_file(str(file_d))
    assert violations_d == []

def test_compliance_fuzz_markdown(tmp_path):
    """Fuzz compliance checker with random markdown characters to ensure no unexpected crashes."""
    alphabet = ['[', ']', '#', '\n', '-', ' ', '>', '\ufeff', '\r', 'a', '`', '~', '<', '!', '-', '%', ':', '\\', 't', 'g', 's', 'o', 'f', 'n', 'e', 'w', 'r', 'i', 'p', 'c', '{', '}', '\t']
    random.seed(42)
    
    for iteration in range(100):
        # Generate random markdown-like text
        length = random.randint(100, 2000)
        chars = [random.choice(alphabet) for _ in range(length)]
        
        if random.random() < 0.5:
            header = "---\ntags:\n  - fact_note\n---\n"
        else:
            header = ""
            
        fuzzed_content = header + "".join(chars)
        file_path = tmp_path / f"fuzzed_{iteration}.md"
        with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(fuzzed_content)
            
        try:
            # We don't assert violations count since random inputs can be valid or invalid.
            # We only assert that check_file never raises an exception.
            check_file(str(file_path))
        except Exception as e:
            pytest.fail(f"compliance_checker.check_file crashed during fuzzing on input:\n{fuzzed_content!r}\nError: {e}")

def test_compliance_extremely_long_lines(tmp_path):
    """Verify that checker can handle lines with extremely large lengths without crashing or backtracking."""
    file_path = tmp_path / "long_lines.md"
    
    # 1,000,000 characters line (containing a header without space at the end)
    long_line_violation = "a" * 1000000 + " #Heading"
    # 1,000,000 characters line (containing a correct header)
    long_line_correct = "b" * 1000000 + " # Heading"
    
    content = f"""---
tags:
  - fact_note
---
{long_line_violation}

{long_line_correct}
"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    start_time = time.perf_counter()
    violations = check_file(str(file_path))
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    assert duration < 2.0, f"Catastrophic backtracking or slow regex: took {duration:.2f} seconds"
    # Note: " #Heading" and " # Heading" will be processed after normalization.
    # Let's check if the violation was found.
    # Since stripped = long_line_violation.strip(), which is "a... #Heading".
    # Since it starts with "a" (not "#"), it won't be flagged as a heading format violation.
    # This is correct. Let's make sure it doesn't crash.
    assert isinstance(violations, list)
