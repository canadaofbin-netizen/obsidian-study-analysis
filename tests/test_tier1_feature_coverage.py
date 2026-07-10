import os
import sys
import json
import time
import pytest
import shutil
import pypdf
import yaml
from unittest.mock import patch, MagicMock

# Ensure scripts directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from download_papers import download_arxiv_papers
from extract_facts import main as run_extraction
from run_debate import main as run_debate, load_personas
from compliance_checker import check_file, check_vault, main as run_compliance
from run_pipeline import main as run_pipeline

# ---------------------------------------------------------
# Feature 1: Paper Ingestion (download_papers.py)
# ---------------------------------------------------------

def test_ingestion_list(tmp_path):
    """F1.1: Fetching and parsing metadata XML correctly."""
    output_dir = tmp_path / "raw_sources"
    registry_path = output_dir / "download_registry.json"
    
    download_arxiv_papers(
        query='all:AI',
        max_results=2,
        output_dir=str(output_dir),
        registry_path=str(registry_path)
    )
    
    assert os.path.exists(registry_path)
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
        
    assert "2401.00001" in registry
    assert "2401.00002" in registry
    assert registry["2401.00001"]["title"] == "AI Agents in Education"

def test_ingestion_download(tmp_path):
    """F1.2: Successfully downloading PDFs."""
    output_dir = tmp_path / "raw_sources"
    registry_path = output_dir / "download_registry.json"
    
    download_arxiv_papers(
        query='all:AI',
        max_results=2,
        output_dir=str(output_dir),
        registry_path=str(registry_path)
    )
    
    pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
    assert len(pdf_files) == 2
    assert "AI Agents in Education.pdf" in pdf_files
    
    # Check pdf contents contain the mock pdf signature
    pdf_path = output_dir / "AI Agents in Education.pdf"
    with open(pdf_path, 'rb') as f:
        content = f.read()
    assert b"%PDF-1.4" in content

def test_ingestion_skip_duplicates(tmp_path):
    """F1.3: Skipping download if file exists."""
    output_dir = tmp_path / "raw_sources"
    registry_path = output_dir / "download_registry.json"
    
    # First run
    download_arxiv_papers(
        query='all:AI',
        max_results=2,
        output_dir=str(output_dir),
        registry_path=str(registry_path)
    )
    
    with patch('urllib.request.urlretrieve') as mock_retrieve:
        # Second run
        download_arxiv_papers(
            query='all:AI',
            max_results=2,
            output_dir=str(output_dir),
            registry_path=str(registry_path)
        )
        # Since files and registry entries exist, urlretrieve should not be called
        mock_retrieve.assert_not_called()

def test_ingestion_directory_creation(tmp_path):
    """F1.4: Creating target folder if absent."""
    nested_output_dir = tmp_path / "non_existent_folder" / "nested_raw_sources"
    registry_path = nested_output_dir / "download_registry.json"
    
    assert not os.path.exists(nested_output_dir)
    
    download_arxiv_papers(
        query='all:AI',
        max_results=1,
        output_dir=str(nested_output_dir),
        registry_path=str(registry_path)
    )
    
    assert os.path.exists(nested_output_dir)

def test_ingestion_throttling(tmp_path):
    """F1.5: Delaying between downloads."""
    output_dir = tmp_path / "raw_sources"
    registry_path = output_dir / "download_registry.json"
    
    with patch('time.sleep') as mock_sleep:
        download_arxiv_papers(
            query='all:AI',
            max_results=2,
            output_dir=str(output_dir),
            registry_path=str(registry_path)
        )
        
        # Throttling is 3 seconds, called after each successful download
        # Since there are 2 papers, sleep(3) should be called twice.
        assert mock_sleep.call_count >= 2
        mock_sleep.assert_any_call(3)

# ---------------------------------------------------------
# Feature 2: Fact Extraction (extract_facts.py)
# ---------------------------------------------------------

def test_extraction_read_pdf(tmp_path):
    """F2.1: Reading PDF text input."""
    pdf_path = tmp_path / "Test.pdf"
    with open(pdf_path, 'wb') as f:
        f.write(b"%PDF-1.4\n%%EOF")
        
    with patch('pypdf.PdfReader') as mock_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is a PDF text with hypothesis."
        mock_reader.return_value.pages = [mock_page]
        
        from extract_facts import extract_text_from_pdf
        text = extract_text_from_pdf(str(pdf_path))
        
        assert "hypothesis" in text
        mock_reader.assert_called_once_with(str(pdf_path))

def test_extraction_call_llm(tmp_path):
    """F2.2: Invoking mock LLM API with correct arguments."""
    pdf_dir = tmp_path / "raw_sources"
    os.makedirs(pdf_dir)
    pdf_path = pdf_dir / "AI Agents in Education.pdf"
    with open(pdf_path, 'wb') as f:
        f.write(b"%PDF-1.4\n%%EOF")
        
    output_dir = tmp_path / "wiki"
    state_path = pdf_dir / "processed_papers.json"
    
    import google.generativeai as genai
    with patch.object(genai.GenerativeModel, 'generate_content', wraps=genai.GenerativeModel("gemini-pro").generate_content) as mock_gen:
        run_extraction([
            '--pdf-dir', str(pdf_dir),
            '--output-dir', str(output_dir),
            '--state-path', str(state_path)
        ])
        
        assert mock_gen.called
        args, kwargs = mock_gen.call_args
        prompt = args[0]
        assert "AI Agents in Education" in prompt

def test_extraction_write_file(tmp_path):
    """F2.3: Generating Fact note markdown file."""
    pdf_dir = tmp_path / "raw_sources"
    os.makedirs(pdf_dir)
    pdf_path = pdf_dir / "AI Agents in Education.pdf"
    with open(pdf_path, 'wb') as f:
        f.write(b"%PDF-1.4\n%%EOF")
        
    output_dir = tmp_path / "wiki"
    state_path = pdf_dir / "processed_papers.json"
    
    run_extraction([
        '--pdf-dir', str(pdf_dir),
        '--output-dir', str(output_dir),
        '--state-path', str(state_path)
    ])
    
    fact_note = output_dir / "Fact - AI Agents in Education.md"
    assert os.path.exists(fact_note)

def test_extraction_yaml_format(tmp_path):
    """F2.4: Fact note YAML structure."""
    pdf_dir = tmp_path / "raw_sources"
    os.makedirs(pdf_dir)
    pdf_path = pdf_dir / "AI Agents in Education.pdf"
    with open(pdf_path, 'wb') as f:
        f.write(b"%PDF-1.4\n%%EOF")
        
    output_dir = tmp_path / "wiki"
    state_path = pdf_dir / "processed_papers.json"
    
    run_extraction([
        '--pdf-dir', str(pdf_dir),
        '--output-dir', str(output_dir),
        '--state-path', str(state_path)
    ])
    
    fact_note = output_dir / "Fact - AI Agents in Education.md"
    with open(fact_note, 'r', encoding='utf-8') as f:
        content = f.read()
        
    lines = content.splitlines()
    assert lines[0] == '---'
    
    closing_idx = -1
    for idx, line in enumerate(lines[1:], start=1):
        if line == '---':
            closing_idx = idx
            break
    assert closing_idx != -1
    
    yaml_block = "\n".join(lines[1:closing_idx])
    parsed = yaml.safe_load(yaml_block)
    assert isinstance(parsed, dict)
    assert 'tags' in parsed
    assert isinstance(parsed['tags'], list)

def test_extraction_sections(tmp_path):
    """F2.5: Fact note contents (hypothesis, methodology, empirical findings)."""
    pdf_dir = tmp_path / "raw_sources"
    os.makedirs(pdf_dir)
    pdf_path = pdf_dir / "AI Agents in Education.pdf"
    with open(pdf_path, 'wb') as f:
        f.write(b"%PDF-1.4\n%%EOF")
        
    output_dir = tmp_path / "wiki"
    state_path = pdf_dir / "processed_papers.json"
    
    run_extraction([
        '--pdf-dir', str(pdf_dir),
        '--output-dir', str(output_dir),
        '--state-path', str(state_path)
    ])
    
    fact_note = output_dir / "Fact - AI Agents in Education.md"
    with open(fact_note, 'r', encoding='utf-8') as f:
        content = f.read()
        
    assert "Explicit Hypotheses & Core Arguments" in content or "Hypothesis 1" in content
    assert "Methodologies" in content
    assert "Empirical & Technical Findings" in content

# ---------------------------------------------------------
# Feature 3: Colosseum Debate (run_debate.py)
# ---------------------------------------------------------

def test_debate_personas():
    """F3.1: Loading agent personas."""
    personas = load_personas()
    assert isinstance(personas, dict)
    assert "Meta-Analyst" in personas
    assert "Devil's Advocate" in personas
    assert "Causal Inference Specialist" in personas

def test_debate_simulation(tmp_path):
    """F3.2: Debate exchanges between agents."""
    wiki_dir = tmp_path / "wiki"
    os.makedirs(wiki_dir)
    with open(wiki_dir / "Fact - AI Agents in Education.md", 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - fact_note\n---\n# Fact Note: AI Agents in Education\n")
    with open(wiki_dir / "Fact - Reinforcement Learning for Autonomous Cars.md", 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - fact_note\n---\n# Fact Note: Reinforcement Learning for Autonomous Cars\n")
        
    import google.generativeai as genai
    with patch.object(genai.GenerativeModel, 'generate_content', wraps=genai.GenerativeModel("gemini-pro").generate_content) as mock_gen:
        run_debate([
            '--wiki-dir', str(wiki_dir),
            '--synthesis-dir', str(wiki_dir / "synthesis"),
            '--concepts-dir', str(wiki_dir / "concepts")
        ])
        assert mock_gen.called
        args, kwargs = mock_gen.call_args
        prompt = args[0]
        assert "Meta-Analyst" in prompt
        assert "Devil's Advocate" in prompt
        assert "Causal Inference Specialist" in prompt

def test_debate_synthesis_file(tmp_path):
    """F3.3: Generating Relationship note."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    
    os.makedirs(wiki_dir)
    with open(wiki_dir / "Fact - AI Agents in Education.md", 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - fact_note\n---\n# Fact Note: AI Agents in Education\n")
    with open(wiki_dir / "Fact - Dummy.md", 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - fact_note\n---\n# Fact Note: Dummy\n")
        
    run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(wiki_dir / "concepts")
    ])
    
    import glob
    relationship_files = glob.glob(str(synthesis_dir / "Relationship - *.md"))
    assert len(relationship_files) == 1
    assert os.path.exists(relationship_files[0])

def test_debate_concept_hubs(tmp_path):
    """F3.4: Generating/updating concept hub nodes."""
    wiki_dir = tmp_path / "wiki"
    concepts_dir = wiki_dir / "concepts"
    
    os.makedirs(wiki_dir)
    with open(wiki_dir / "Fact - AI Agents in Education.md", 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - fact_note\n---\n# Fact Note: AI Agents in Education\n")
    with open(wiki_dir / "Fact - Dummy.md", 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - fact_note\n---\n# Fact Note: Dummy\n")
        
    run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(wiki_dir / "synthesis"),
        '--concepts-dir', str(concepts_dir)
    ])
    
    import glob
    concept_files = glob.glob(str(concepts_dir / "Concept - *.md"))
    assert len(concept_files) == 1
    concept_file = concept_files[0]
    assert os.path.exists(concept_file)
    
    concept_title = os.path.basename(concept_file).replace("Concept - ", "").replace(".md", "")
    with open(wiki_dir / "Fact - AI Agents in Education.md", 'r', encoding='utf-8') as f:
        fact_content = f.read()
    assert f"[[Concept - {concept_title}]]" in fact_content

def test_debate_wikilinks(tmp_path):
    """F3.5: Referencing fact notes with [[wikilinks]] inside relationship notes."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    
    os.makedirs(wiki_dir)
    with open(wiki_dir / "Fact - AI Agents in Education.md", 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - fact_note\n---\n# Fact Note: AI Agents in Education\n")
    with open(wiki_dir / "Fact - Dummy.md", 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - fact_note\n---\n# Fact Note: Dummy\n")
        
    run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(wiki_dir / "concepts")
    ])
    
    import glob
    relationship_files = glob.glob(str(synthesis_dir / "Relationship - *.md"))
    assert len(relationship_files) == 1
    relationship_file = relationship_files[0]
    with open(relationship_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    assert "[[Fact - AI Agents in Education]]" in content

# ---------------------------------------------------------
# Feature 4: Obsidian Compliance Validator
# ---------------------------------------------------------

def test_compliance_yaml(tmp_path):
    """F4.1: Validating YAML syntax (starting and ending with '---', tags as list)."""
    valid_file = tmp_path / "valid.md"
    with open(valid_file, 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - test_tag\n---\n# Title\n")
    violations = check_file(str(valid_file))
    assert len(violations) == 0
    
    invalid_file1 = tmp_path / "invalid1.md"
    with open(invalid_file1, 'w', encoding='utf-8') as f:
        f.write("tags:\n  - test_tag\n---\n# Title\n")
    violations = check_file(str(invalid_file1))
    assert any("YAML boundary" in v for v in violations)
    
    invalid_file2 = tmp_path / "invalid2.md"
    with open(invalid_file2, 'w', encoding='utf-8') as f:
        f.write("---\ntags: test_tag\n---\n# Title\n")
    violations = check_file(str(invalid_file2))
    assert any("tags" in v.lower() and "list" in v.lower() for v in violations)

def test_compliance_wikilinks(tmp_path):
    """F4.2: Validating that wikilinks exist and are syntactically correct."""
    valid_file = tmp_path / "valid.md"
    with open(valid_file, 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - tag\n---\nThis is a [[WikiLink]].\n")
    violations = check_file(str(valid_file))
    assert len(violations) == 0
    
    invalid_file1 = tmp_path / "invalid1.md"
    with open(invalid_file1, 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - tag\n---\nThis is a [[Mismatched bracket.\n")
    violations = check_file(str(invalid_file1))
    assert any("Mismatched" in v for v in violations)
    
    invalid_file2 = tmp_path / "invalid2.md"
    with open(invalid_file2, 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - tag\n---\nThis is an [[]] empty wikilink.\n")
    violations = check_file(str(invalid_file2))
    assert any("Empty wikilink" in v for v in violations)

def test_compliance_rendering(tmp_path):
    """F4.3: Verification of document rendering integrity."""
    invalid_file = tmp_path / "invalid.md"
    with open(invalid_file, 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - tag\n---\n#TitleWithoutSpace\n")
    violations = check_file(str(invalid_file))
    assert any("Heading format" in v for v in violations)

def test_compliance_report_violations(tmp_path):
    """F4.4: Exits with non-zero code on violations."""
    vault_dir = tmp_path / "vault"
    os.makedirs(vault_dir)
    
    invalid_file = vault_dir / "invalid.md"
    with open(invalid_file, 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - tag\n---\n#Title\n")
        
    with pytest.raises(SystemExit) as excinfo:
        run_compliance(['--vault-dir', str(vault_dir)])
    assert excinfo.value.code != 0

def test_compliance_clean_run(tmp_path):
    """F4.5: Exits with zero code on fully compliant vault."""
    vault_dir = tmp_path / "vault"
    os.makedirs(vault_dir)
    
    valid_file = vault_dir / "valid.md"
    with open(valid_file, 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - tag\n---\n# Valid Title\n")
        
    with pytest.raises(SystemExit) as excinfo:
        run_compliance(['--vault-dir', str(vault_dir)])
    assert excinfo.value.code == 0

# ---------------------------------------------------------
# Feature 5: End-to-End Pipeline Integration (run_pipeline.py)
# ---------------------------------------------------------

def test_pipeline_runner_cli(tmp_path):
    """F5.1: Accepts correct arguments/parameters."""
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    log_path = tmp_path / "pipeline.log"
    
    run_pipeline([
        '--query', 'all:AI',
        '--max-results', '1',
        '--pdf-dir', str(pdf_dir),
        '--wiki-dir', str(wiki_dir),
        '--log-path', str(log_path)
    ])
    
    assert os.path.exists(log_path)

def test_pipeline_sequence(tmp_path):
    """F5.2: Ingest -> Extract -> Debate -> Compliance executed in order."""
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    log_path = tmp_path / "pipeline.log"
    
    run_pipeline([
        '--query', 'all:AI',
        '--max-results', '2',
        '--pdf-dir', str(pdf_dir),
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir),
        '--log-path', str(log_path)
    ])
    
    # Verify sequence output artifacts
    # 1. Ingest
    assert len(os.listdir(pdf_dir)) >= 2
    # 2. Extract
    assert os.path.exists(wiki_dir / "Fact - AI Agents in Education.md")
    assert os.path.exists(wiki_dir / "Fact - Reinforcement Learning for Autonomous Cars.md")
    # 3. Debate
    import glob
    assert len(glob.glob(str(synthesis_dir / "Relationship - *.md"))) == 1
    assert len(glob.glob(str(concepts_dir / "Concept - *.md"))) == 1

def test_pipeline_empty_run(tmp_path):
    """F5.3: Correct behavior when no new papers are found."""
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    log_path = tmp_path / "pipeline.log"
    registry_path = pdf_dir / "download_registry.json"
    
    os.makedirs(pdf_dir)
    registry = {
        "2401.00001": {"title": "AI Agents in Education", "filename": "AI Agents in Education.pdf"},
        "2401.00002": {"title": "Reinforcement Learning for Autonomous Cars", "filename": "Reinforcement Learning for Autonomous Cars.pdf"}
    }
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f)
        
    with open(pdf_dir / "AI Agents in Education.pdf", 'wb') as f:
        f.write(b"%PDF-1.4\n%%EOF")
    with open(pdf_dir / "Reinforcement Learning for Autonomous Cars.pdf", 'wb') as f:
        f.write(b"%PDF-1.4\n%%EOF")
        
    run_pipeline([
        '--query', 'all:AI',
        '--max-results', '2',
        '--pdf-dir', str(pdf_dir),
        '--wiki-dir', str(wiki_dir),
        '--registry-path', str(registry_path),
        '--log-path', str(log_path)
    ])
    
    fact_file = wiki_dir / "Fact - AI Agents in Education.md"
    assert os.path.exists(fact_file)
    mtime_before = os.path.getmtime(fact_file)
    
    time.sleep(0.1)
    run_pipeline([
        '--query', 'all:AI',
        '--max-results', '2',
        '--pdf-dir', str(pdf_dir),
        '--wiki-dir', str(wiki_dir),
        '--registry-path', str(registry_path),
        '--log-path', str(log_path)
    ])
    
    mtime_after = os.path.getmtime(fact_file)
    assert mtime_before == mtime_after

def test_pipeline_state_tracking(tmp_path):
    """F5.4: Avoids re-processing already processed papers."""
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    log_path = tmp_path / "pipeline.log"
    state_path = pdf_dir / "processed_papers.json"
    
    os.makedirs(pdf_dir)
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(["AI Agents in Education.pdf", "Reinforcement Learning for Autonomous Cars.pdf"], f)
        
    run_pipeline([
        '--query', 'all:AI',
        '--max-results', '2',
        '--pdf-dir', str(pdf_dir),
        '--wiki-dir', str(wiki_dir),
        '--state-path', str(state_path),
        '--log-path', str(log_path)
    ])
    
    assert not os.path.exists(wiki_dir / "Fact - AI Agents in Education.md")
    assert not os.path.exists(wiki_dir / "Fact - Reinforcement Learning for Autonomous Cars.md")

def test_pipeline_logging(tmp_path):
    """F5.5: Writes execution log files."""
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    log_path = tmp_path / "pipeline.log"
    
    run_pipeline([
        '--query', 'all:AI',
        '--max-results', '1',
        '--pdf-dir', str(pdf_dir),
        '--wiki-dir', str(wiki_dir),
        '--log-path', str(log_path)
    ])
    
    assert os.path.exists(log_path)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        
    assert "Pipeline started" in log_content
    assert "Step 1: Running Paper Ingestion" in log_content
    assert "Pipeline executed and finished successfully" in log_content
