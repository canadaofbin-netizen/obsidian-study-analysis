import os
import sys
import json
import re
import pytest
import yaml
from unittest.mock import patch, MagicMock

# Ensure scripts directory is in python path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if "explorer_tier3_3" in _current_dir:
    _project_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
    _scripts_dir = os.path.join(_project_root, "obsidianfolder", "scripts")
else:
    _scripts_dir = os.path.abspath(os.path.join(_current_dir, "..", "scripts"))

if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from download_papers import download_arxiv_papers
from extract_facts import main as run_extraction
from run_debate import main as run_debate
from compliance_checker import check_vault, check_file
from run_pipeline import main as run_pipeline


# ---------------------------------------------------------
# Test 1: Ingestion -> Fact Extraction
# ---------------------------------------------------------
def test_ingestion_to_fact_extraction(tmp_path):
    """
    Test 1: Ingestion -> Fact Extraction
    Verify that ingested PDFs are successfully scanned and extracted.
    """
    pdf_dir = tmp_path / "raw_sources"
    registry_path = pdf_dir / "download_registry.json"
    wiki_dir = tmp_path / "wiki"
    state_path = pdf_dir / "processed_papers.json"

    # Step 1: Run Ingestion
    download_arxiv_papers(
        query='all:"AI Agents"',
        max_results=2,
        output_dir=str(pdf_dir),
        registry_path=str(registry_path)
    )

    # Verify ingestion outputs
    assert os.path.exists(registry_path)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    assert len(pdf_files) == 2
    assert "AI Agents in Education.pdf" in pdf_files
    assert "Reinforcement Learning for Autonomous Cars.pdf" in pdf_files

    # Step 2: Run Fact Extraction
    ret = run_extraction([
        '--pdf-dir', str(pdf_dir),
        '--output-dir', str(wiki_dir),
        '--state-path', str(state_path)
    ])
    assert ret == 0

    # Verify fact extraction outputs
    assert os.path.exists(state_path)
    with open(state_path, 'r', encoding='utf-8') as f:
        state_data = json.load(f)
    assert "AI Agents in Education.pdf" in state_data
    assert "Reinforcement Learning for Autonomous Cars.pdf" in state_data

    fact_notes = [f for f in os.listdir(wiki_dir) if f.startswith("Fact - ") and f.endswith(".md")]
    assert len(fact_notes) == 2
    assert "Fact - AI Agents in Education.md" in fact_notes
    assert "Fact - Reinforcement Learning for Autonomous Cars.md" in fact_notes

    # Check contents of one fact note
    fact_note_path = wiki_dir / "Fact - AI Agents in Education.md"
    with open(fact_note_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert "tags:" in content
    assert "fact_note" in content
    assert "# Fact Note: AI Agents in Education" in content


# ---------------------------------------------------------
# Test 2: Ingestion -> Pipeline Runner
# ---------------------------------------------------------
def test_ingestion_to_pipeline_runner(tmp_path, mock_urllib_urlopen):
    """
    Test 2: Ingestion -> Pipeline Runner
    Verify that pipeline runner handles ingestion successfully and propagates download parameters.
    """
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    state_path = pdf_dir / "processed_papers.json"
    registry_path = pdf_dir / "download_registry.json"
    log_path = tmp_path / "pipeline.log"

    # Reset mock urlopen to clear call history
    mock_urllib_urlopen.reset_mock()

    # Run Pipeline with custom query and max results
    run_pipeline([
        '--query', 'custom_query_testing',
        '--max-results', '5',
        '--pdf-dir', str(pdf_dir),
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir),
        '--state-path', str(state_path),
        '--registry-path', str(registry_path),
        '--log-path', str(log_path)
    ])

    # Verify query and max-results propagation via urlopen call arguments
    assert mock_urllib_urlopen.call_count >= 1
    called_url = mock_urllib_urlopen.call_args[0][0]
    assert "search_query=custom_query_testing" in called_url
    assert "max_results=5" in called_url

    # Verify files were downloaded and registered
    assert os.path.exists(registry_path)
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    assert "2401.00001" in registry
    assert "2401.00002" in registry

    # Verify pipeline completed successfully generating log file
    assert os.path.exists(log_path)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    assert "Pipeline started." in log_content
    assert "Step 1: Running Paper Ingestion..." in log_content
    assert "Step 4: Running Compliance Validation..." in log_content
    assert "Pipeline executed and finished successfully." in log_content


# ---------------------------------------------------------
# Test 3: Fact Extraction -> Colosseum Debate
# ---------------------------------------------------------
def test_fact_extraction_to_colosseum_debate(tmp_path):
    """
    Test 3: Fact Extraction -> Colosseum Debate
    Verify that extracted fact notes are referenced correctly in debate transcripts with proper Obsidian wikilinks.
    """
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    state_path = pdf_dir / "processed_papers.json"

    # Pre-populate two PDF files to extract
    os.makedirs(pdf_dir, exist_ok=True)
    for title in ["AI Agents in Education", "Reinforcement Learning for Autonomous Cars"]:
        pdf_path = pdf_dir / f"{title}.pdf"
        with open(pdf_path, 'wb') as f:
            f.write(b"%PDF-1.4\n%%EOF")

    # Run extraction
    ret_extraction = run_extraction([
        '--pdf-dir', str(pdf_dir),
        '--output-dir', str(wiki_dir),
        '--state-path', str(state_path)
    ])
    assert ret_extraction == 0

    # Verify extraction outputs
    fact1_path = wiki_dir / "Fact - AI Agents in Education.md"
    fact2_path = wiki_dir / "Fact - Reinforcement Learning for Autonomous Cars.md"
    assert os.path.exists(fact1_path)
    assert os.path.exists(fact2_path)

    # Run Debate
    ret_debate = run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir)
    ])
    assert ret_debate == 0

    # Verify Relationship Note references source fact notes with Obsidian wikilinks
    relationship_files = os.listdir(synthesis_dir)
    assert len(relationship_files) >= 1
    relationship_note_path = synthesis_dir / relationship_files[0]
    
    with open(relationship_note_path, 'r', encoding='utf-8') as f:
        rel_content = f.read()

    # Frontmatter and body check for WikiLinks
    assert "source_papers:" in rel_content
    assert "[[Fact - AI Agents in Education]]" in rel_content
    assert "[[Fact - Reinforcement Learning for Autonomous Cars]]" in rel_content

    # Verify Concept Hub references source fact notes with Obsidian wikilinks
    concept_files = os.listdir(concepts_dir)
    assert len(concept_files) >= 1
    concept_hub_path = concepts_dir / concept_files[0]

    with open(concept_hub_path, 'r', encoding='utf-8') as f:
        concept_content = f.read()

    assert "[[Fact - AI Agents in Education]]" in concept_content
    assert "[[Fact - Reinforcement Learning for Autonomous Cars]]" in concept_content

    # Verify bidirectional updates inside Fact Notes
    with open(fact1_path, 'r', encoding='utf-8') as f:
        f1_content = f.read()
    assert "[[Concept -" in f1_content
    assert "[[Relationship -" in f1_content


# ---------------------------------------------------------
# Test 4: Colosseum Debate -> Compliance Validator
# ---------------------------------------------------------
def test_colosseum_debate_to_compliance_validator(tmp_path):
    """
    Test 4: Colosseum Debate -> Compliance Validator
    Verify that run_debate output meets YAML and WikiLinks compliance rules.
    """
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"

    os.makedirs(wiki_dir, exist_ok=True)
    os.makedirs(synthesis_dir, exist_ok=True)
    os.makedirs(concepts_dir, exist_ok=True)

    # Pre-populate two valid Fact Notes (required by run_debate script to find papers)
    fact1_content = """---
tags:
  - fact_note
---
# Fact Note: AI Agents in Education
* **Hypothesis 1:** AI agents optimize learning pathways.
"""
    fact2_content = """---
tags:
  - fact_note
---
# Fact Note: Reinforcement Learning for Autonomous Cars
* **Hypothesis 1:** Reinforcement learning controls steering.
"""
    with open(wiki_dir / "Fact - AI Agents in Education.md", "w", encoding="utf-8") as f:
        f.write(fact1_content)
    with open(wiki_dir / "Fact - Reinforcement Learning for Autonomous Cars.md", "w", encoding="utf-8") as f:
        f.write(fact2_content)

    # Run Debate
    ret = run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir)
    ])
    assert ret == 0

    # Scan debate outputs using compliance checker
    violations = check_vault(str(wiki_dir))
    
    # Assert no compliance violations found
    assert not violations, f"Compliance violations found: {violations}"

    # Explicitly check structure of the generated relationship note
    relationship_files = os.listdir(synthesis_dir)
    relationship_note_path = synthesis_dir / relationship_files[0]
    
    with open(relationship_note_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Frontmatter structure assertions
    assert lines[0].strip() == "---"
    
    # Find closing ---
    closing_idx = -1
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_idx = idx
            break
    assert closing_idx != -1
    
    # Parse YAML frontmatter
    yaml_block = "".join(lines[1:closing_idx])
    parsed_yaml = yaml.safe_load(yaml_block)
    
    assert "tags" in parsed_yaml
    assert "relationship_note" in parsed_yaml["tags"]
    assert "debate_transcript" in parsed_yaml["tags"]
    assert "source_papers" in parsed_yaml
    assert isinstance(parsed_yaml["source_papers"], list)
    
    # Check block list format of tags in the file
    tag_lines = []
    in_tags = False
    for line in lines[1:closing_idx]:
        if line.strip().startswith("tags:"):
            in_tags = True
            continue
        if in_tags:
            if line.strip().startswith("-"):
                tag_lines.append(line.strip())
            elif ":" in line:
                in_tags = False
    
    assert len(tag_lines) == 2
    assert tag_lines[0] == "- relationship_note"
    assert tag_lines[1] == "- debate_transcript"


# ---------------------------------------------------------
# Test 5: Fact Extraction -> Compliance Validator
# ---------------------------------------------------------
def test_fact_extraction_to_compliance_validator(tmp_path):
    """
    Test 5: Fact Extraction -> Compliance Validator
    Verify that extract_facts output meets YAML compliance rules.
    """
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    state_path = pdf_dir / "processed_papers.json"

    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = pdf_dir / "AI Agents in Education.pdf"
    with open(pdf_path, 'wb') as f:
        f.write(b"%PDF-1.4\n%%EOF")

    # Run Fact Extraction
    ret = run_extraction([
        '--pdf-dir', str(pdf_dir),
        '--output-dir', str(wiki_dir),
        '--state-path', str(state_path)
    ])
    assert ret == 0

    # Scan extraction outputs using compliance checker
    violations = check_vault(str(wiki_dir))
    
    # Assert no compliance violations found
    assert not violations, f"Compliance violations found: {violations}"

    # Explicitly check structure of the generated fact note
    fact_note_path = wiki_dir / "Fact - AI Agents in Education.md"
    with open(fact_note_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Frontmatter structure assertions
    assert lines[0].strip() == "---"
    
    # Find closing ---
    closing_idx = -1
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_idx = idx
            break
    assert closing_idx != -1
    
    # Parse YAML frontmatter
    yaml_block = "".join(lines[1:closing_idx])
    parsed_yaml = yaml.safe_load(yaml_block)
    
    assert "tags" in parsed_yaml
    assert "fact_note" in parsed_yaml["tags"]
    
    # Check block list format of tags in the file
    tag_lines = []
    in_tags = False
    for line in lines[1:closing_idx]:
        if line.strip().startswith("tags:"):
            in_tags = True
            continue
        if in_tags:
            if line.strip().startswith("-"):
                tag_lines.append(line.strip())
            elif ":" in line:
                in_tags = False
                
    assert len(tag_lines) >= 1
    assert tag_lines[0] == "- fact_note"

    # Verify heading formatting has space after '#'
    heading_found = False
    for line in lines[closing_idx+1:]:
        if line.strip().startswith("#"):
            heading_found = True
            assert line.startswith("# ") or line.startswith("## ") or line.startswith("### ")
            assert not re.match(r'^#{1,6}[^# \t]', line.strip())
            
    assert heading_found
