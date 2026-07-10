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
from extract_facts import main as run_extraction
from run_debate import main as run_debate, get_dynamic_personas
from run_pipeline import main as run_pipeline
from compliance_checker import check_vault

# ---------------------------------------------------------
# Scenario 1: Multi-Paper Pipeline Run (E2E Integration)
# ---------------------------------------------------------

def test_scenario1_multi_paper_pipeline_run(tmp_path, mock_time_sleep):
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    state_path = pdf_dir / "processed_papers.json"
    registry_path = pdf_dir / "download_registry.json"
    log_path = tmp_path / "pipeline.log"
    
    mock_xml = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Quantum Key Distribution and Agents</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.00003" />
        <id>http://arxiv.org/abs/2401.00003v1</id>
      </entry>
      <entry>
        <title>Deep Neural Networks for Autonomous Robotics</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.00004" />
        <id>http://arxiv.org/abs/2401.00004v1</id>
      </entry>
      <entry>
        <title>Algorithmic Game Theory in Multi-Agent Systems</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.00005" />
        <id>http://arxiv.org/abs/2401.00005v1</id>
      </entry>
    </feed>
    """
    
    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def custom_llm_response(prompt_text):
        prompt_lower = prompt_text.lower()
        if "debate" in prompt_lower or "colosseum" in prompt_lower:
            return """---
tags:
  - relationship_node
  - debate_transcript
source_papers:
  - "[[Fact - Quantum Key Distribution and Agents]]"
  - "[[Fact - Deep Neural Networks for Autonomous Robotics]]"
  - "[[Fact - Algorithmic Game Theory in Multi-Agent Systems]]"
---
# Synthesized Debate: Multi-Agent Convergence

**Meta-Analyst:** Welcome to the debate.
**Quantum Physicist:** Analyzed quantum cryptography protocols.
**Robotics & Control Engineer:** Focused on autonomous robot control loops.
**Algorithmic Game Theorist:** Discussed utility equilibrium.

The core connection is the application of game theory.
"""
        elif "quantum" in prompt_lower:
            return """---
tags:
  - fact_note
---
# Fact Note: Quantum Key Distribution and Agents
Contains quantum mechanics concepts.
"""
        elif "robot" in prompt_lower:
            return """---
tags:
  - fact_note
---
# Fact Note: Deep Neural Networks for Autonomous Robotics
Contains autonomous robotics principles.
"""
        else:
            return """---
tags:
  - fact_note
---
# Fact Note: Algorithmic Game Theory in Multi-Agent Systems
Contains multi-agent game theory.
"""

    import google.generativeai as genai

    with patch('urllib.request.urlopen', return_value=MockHTTPResponse(mock_xml.encode('utf-8'))), \
         patch.object(genai.GenerativeModel, 'generate_content') as mock_generate:
        
        # Configure model mock
        mock_generate.side_effect = lambda prompt, *args, **kwargs: MagicMock(text=custom_llm_response(str(prompt)))
        
        # Execute pipeline
        run_pipeline([
            '--query', 'quantum robotics game theory',
            '--max-results', '3',
            '--pdf-dir', str(pdf_dir),
            '--wiki-dir', str(wiki_dir),
            '--synthesis-dir', str(synthesis_dir),
            '--concepts-dir', str(concepts_dir),
            '--state-path', str(state_path),
            '--registry-path', str(registry_path),
            '--log-path', str(log_path)
        ])
        
    # Assertions
    # 1. Ingestion: Downloaded files on disk and updated registry
    assert os.path.exists(registry_path)
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    assert "2401.00003" in registry
    assert "2401.00004" in registry
    assert "2401.00005" in registry
    
    # 2. Fact Extraction: Processed papers state file created and notes saved
    assert os.path.exists(state_path)
    with open(state_path, 'r', encoding='utf-8') as f:
        processed_state = json.load(f)
    assert len(processed_state) == 3
    
    fact_notes = [f for f in os.listdir(wiki_dir) if f.startswith("Fact - ") and f.endswith(".md")]
    assert len(fact_notes) == 3
    
    # 3. Debate: Synthesized relationship note is present and persona active
    relationship_files = os.listdir(synthesis_dir)
    assert len(relationship_files) >= 1
    rel_note_path = synthesis_dir / relationship_files[0]
    with open(rel_note_path, 'r', encoding='utf-8') as f:
        rel_content = f.read()
    assert "Quantum Physicist:" in rel_content
    assert "Robotics & Control Engineer:" in rel_content
    assert "Algorithmic Game Theorist:" in rel_content
    
    # 4. Bidirectional updates inside Fact Notes occurred
    fact_note_sample = wiki_dir / "Fact - Quantum Key Distribution and Agents.md"
    with open(fact_note_sample, 'r', encoding='utf-8') as f:
        fact_content = f.read()
    assert "[[Concept - Game Theory]]" in fact_content
    assert "[[Relationship - " in fact_content
    
    # 5. Log verification
    assert os.path.exists(log_path)
    with open(log_path, 'r', encoding='utf-8') as f:
        log_text = f.read()
    assert "Step 1: Running Paper Ingestion..." in log_text
    assert "Step 2: Running Fact Extraction..." in log_text
    assert "Step 3: Running Colosseum Debate..." in log_text
    assert "Step 4: Running Compliance Validation..." in log_text
    assert "Pipeline executed and finished successfully." in log_text

# ---------------------------------------------------------
# Scenario 2: Vault Incremental Scan (Duplicate Avoidance)
# ---------------------------------------------------------

def test_scenario2_vault_incremental_scan(tmp_path, mock_urllib_urlretrieve, mock_time_sleep):
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    registry_path = pdf_dir / "download_registry.json"
    state_path = pdf_dir / "processed_papers.json"
    
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(wiki_dir, exist_ok=True)
    
    # Seed 1: File on disk but NOT in registry
    paper_a_path = pdf_dir / "Paper A.pdf"
    with open(paper_a_path, 'wb') as f:
        f.write(b"%PDF-1.4\n%dummy pdf A\n%%EOF")
        
    # Seed 2 & 3 Registry setup
    registry_data = {
        "2401.90002": {
            "title": "Paper B",
            "filename": "Paper B.pdf",
            "registered_at": "2026-07-10T00:00:00Z"
        },
        "2401.90003": {
            "title": "Paper C",
            "filename": "Paper C.pdf",
            "registered_at": "2026-07-10T00:00:00Z"
        }
    }
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry_data, f)
        
    # Seed 3: File on disk, registered, and marked as processed in state
    paper_c_path = pdf_dir / "Paper C.pdf"
    with open(paper_c_path, 'wb') as f:
        f.write(b"%PDF-1.4\n%dummy pdf C\n%%EOF")
        
    fact_c_path = wiki_dir / "Fact - Paper C.md"
    with open(fact_c_path, 'w', encoding='utf-8') as f:
        f.write("---\ntags:\n  - fact_note\n---\n# Fact Note: Paper C\n### Status: Pre-seeded")
        
    state_data = ["Paper C.pdf"]
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(state_data, f)
        
    # Seed 4: PDF on disk to be extracted
    paper_d_path = pdf_dir / "Paper D.pdf"
    with open(paper_d_path, 'wb') as f:
        f.write(b"%PDF-1.4\n%dummy pdf D\n%%EOF")
        
    # Mock URL Retrieve track downloads
    download_urls = []
    def track_retrieve(url, filename, *args, **kwargs):
        download_urls.append(url)
        with open(filename, 'wb') as f:
            f.write(b"%PDF-1.4\n%downloaded\n%%EOF")
        return (filename, MagicMock())
    mock_urllib_urlretrieve.side_effect = track_retrieve

    # Mock URLOpen to return arXiv feed containing Paper A (ID: 2401.90001), Paper B, Paper C
    mock_xml = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Paper A</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.90001" />
        <id>http://arxiv.org/abs/2401.90001v1</id>
      </entry>
      <entry>
        <title>Paper B</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.90002" />
        <id>http://arxiv.org/abs/2401.90002v1</id>
      </entry>
      <entry>
        <title>Paper C</title>
        <link title="pdf" href="http://export.arxiv.org/pdf/2401.90003" />
        <id>http://arxiv.org/abs/2401.90003v1</id>
      </entry>
    </feed>
    """
    
    class MockHTTPResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, exc, val, tb):
            pass

    with patch('urllib.request.urlopen', return_value=MockHTTPResponse(mock_xml.encode('utf-8'))):
        # 1. Run Ingestion
        download_arxiv_papers(
            query="all:AI",
            max_results=3,
            output_dir=str(pdf_dir),
            registry_path=str(registry_path)
        )
        
    # Ingestion Assertions:
    # - Paper A (disk exists) should NOT be downloaded but registered
    # - Paper B (registry exists, disk missing) should NOT be downloaded (due to in_registry check)
    # - Paper C (registry and disk exist) should NOT be downloaded
    assert len(download_urls) == 0, f"Expected 0 downloads, got: {download_urls}"
    
    with open(registry_path, 'r', encoding='utf-8') as f:
        updated_registry = json.load(f)
    assert "2401.90001" in updated_registry # Paper A registered
    assert "2401.90002" in updated_registry # Paper B remains
    assert "2401.90003" in updated_registry # Paper C remains
    
    # 2. Run Extraction
    ret = run_extraction([
        '--pdf-dir', str(pdf_dir),
        '--output-dir', str(wiki_dir),
        '--state-path', str(state_path)
    ])
    assert ret == 0
    
    # Extraction Assertions:
    # - Paper C should be skipped (Fact - Paper C.md untouched)
    with open(fact_c_path, 'r', encoding='utf-8') as f:
        fact_c_content = f.read()
    assert "### Status: Pre-seeded" in fact_c_content
    
    # - Paper A and Paper D should be extracted
    assert os.path.exists(wiki_dir / "Fact - Paper A.md")
    assert os.path.exists(wiki_dir / "Fact - Paper D.md")
    
    # - State file updated containing A, C, D
    with open(state_path, 'r', encoding='utf-8') as f:
        final_state = json.load(f)
    assert "Paper A.pdf" in final_state
    assert "Paper C.pdf" in final_state
    assert "Paper D.pdf" in final_state
    
    # 3. Test force reprocess option (--force)
    ret_force = run_extraction([
        '--pdf-dir', str(pdf_dir),
        '--output-dir', str(wiki_dir),
        '--state-path', str(state_path),
        '--force'
    ])
    assert ret_force == 0
    
    # Paper C should now be overwritten (custom pre-seeded comment gone)
    with open(fact_c_path, 'r', encoding='utf-8') as f:
        fact_c_content_after = f.read()
    assert "### Status: Pre-seeded" not in fact_c_content_after

# ---------------------------------------------------------
# Scenario 3: Network Error Recovery
# ---------------------------------------------------------

def test_scenario3_network_error_recovery(tmp_path, mock_time_sleep, mock_urllib_urlopen):
    # A. Network Ingestion Transient Retry
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = b"<feed><entry><title>Transient Paper</title><link title='pdf' href='http://export.arxiv.org/pdf/2401.00001' /></entry></feed>"
    
    mock_urllib_urlopen.reset_mock()
    mock_time_sleep.reset_mock()
    mock_urllib_urlopen.side_effect = [
        urllib.error.HTTPError("http://export.arxiv.org", 503, "Service Unavailable", None, None),
        urllib.error.HTTPError("http://export.arxiv.org", 503, "Service Unavailable", None, None),
        mock_resp
    ]
    
    data = make_request_with_retry("http://export.arxiv.org/api/query", is_download=False)
    assert b"Transient Paper" in data
    assert mock_urllib_urlopen.call_count == 3
    assert mock_time_sleep.call_count == 3  # 2 backoffs (3s, 6s) + 1 post-success delay (3s)

    # B. Network Ingestion Persistent Failure
    mock_urllib_urlopen.reset_mock()
    mock_time_sleep.reset_mock()
    mock_urllib_urlopen.side_effect = urllib.error.HTTPError("http://export.arxiv.org", 503, "Service Unavailable", None, None)
    
    with pytest.raises(urllib.error.HTTPError):
        make_request_with_retry("http://export.arxiv.org/api/query", is_download=False)
        
    assert mock_urllib_urlopen.call_count == 3
    assert mock_time_sleep.call_count == 2  # 2 backoffs before reraise

    # C. Extraction PDF Fault Tolerance (Corrupted and Scanned/Empty PDFs)
    pdf_dir = tmp_path / "raw_sources"
    wiki_dir = tmp_path / "wiki"
    state_path = pdf_dir / "processed_papers.json"
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(wiki_dir, exist_ok=True)
    
    (pdf_dir / "corrupted.pdf").write_bytes(b"")
    
    # Scanned PDF (empty text)
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with open(pdf_dir / "scanned.pdf", "wb") as f:
        writer.write(f)
        
    # Valid PDF
    valid_pdf_path = pdf_dir / "valid_paper.pdf"
    with open(valid_pdf_path, "wb") as f:
        f.write(b"%PDF-1.4\n%dummy\n%%EOF")
        
    import google.generativeai as genai
    mock_response = MagicMock()
    mock_response.text = "---\ntags:\n  - fact_note\n---\n# Fact Note: Valid Paper\nHypothesis 1: Success."
    
    def dynamic_pdf_reader(path, *args, **kwargs):
        filename = os.path.basename(path)
        reader = MagicMock()
        reader.is_encrypted = False
        if filename == "corrupted.pdf":
            raise Exception("Corrupt PDF structure")
        elif filename == "scanned.pdf":
            page = MagicMock()
            page.extract_text.return_value = ""
            reader.pages = [page]
        else:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "This is a dummy extracted PDF text that is long enough to pass the length guard check (minimum 100 characters)."
            reader.pages = [mock_page]
        return reader
        
    with patch("pypdf.PdfReader", side_effect=dynamic_pdf_reader), \
         patch.object(genai.GenerativeModel, 'generate_content', return_value=mock_response):
         
        code = run_extraction([
            '--pdf-dir', str(pdf_dir),
            '--output-dir', str(wiki_dir),
            '--state-path', str(state_path)
        ])
        
    assert code == 0
    assert os.path.exists(wiki_dir / "Fact - valid_paper.md")
    assert not os.path.exists(wiki_dir / "Fact - corrupted.md")
    assert not os.path.exists(wiki_dir / "Fact - scanned.md")
    
    with open(state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)
    assert "valid_paper.pdf" in state
    assert "corrupted.pdf" not in state
    assert "scanned.pdf" not in state

    # D. Gemini API Extraction Recovery
    # Reset directories
    for f in os.listdir(pdf_dir):
        os.remove(pdf_dir / f)
    for f in os.listdir(wiki_dir):
        os.remove(wiki_dir / f)
    if os.path.exists(state_path):
        os.remove(state_path)
        
    (pdf_dir / "paper1.pdf").write_bytes(b"%PDF-1.4\n%%EOF")
    (pdf_dir / "paper2.pdf").write_bytes(b"%PDF-1.4\n%%EOF")
    
    mock_model = MagicMock()
    def dynamic_generate(prompt, *args, **kwargs):
        if "paper1.pdf" in str(prompt):
            raise Exception("Quota exceeded / Rate limited")
        resp = MagicMock()
        resp.text = "---\ntags:\n  - fact_note\n---\n# Fact Note: Paper 2"
        return resp
    mock_model.generate_content.side_effect = dynamic_generate
    
    mock_page = MagicMock()
    # Long text (> 100 characters) to pass length check
    mock_page.extract_text.return_value = "Long enough text to pass the 100 character threshold guard for both files. Let's make sure it contains enough characters to satisfy the guards and not get skipped."
    
    with patch("pypdf.PdfReader") as MockReader, \
         patch("google.generativeai.GenerativeModel", return_value=mock_model):
        mock_reader_instance = MagicMock()
        mock_reader_instance.is_encrypted = False
        mock_reader_instance.pages = [mock_page]
        MockReader.return_value = mock_reader_instance
        
        run_extraction([
            '--pdf-dir', str(pdf_dir),
            '--output-dir', str(wiki_dir),
            '--state-path', str(state_path)
        ])
        
    assert os.path.exists(wiki_dir / "Fact - paper2.md")
    assert not os.path.exists(wiki_dir / "Fact - paper1.md")
    
    with open(state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)
    assert "paper2.pdf" in state
    assert "paper1.pdf" not in state

# ---------------------------------------------------------
# Scenario 4: Large Vault Debate
# ---------------------------------------------------------

@patch("google.generativeai.GenerativeModel")
def test_scenario4_large_vault_debate(mock_model_class, tmp_path):
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    os.makedirs(wiki_dir, exist_ok=True)
    
    # Setup 4 papers matching 4 specialized personas
    papers = {
        "Fact - Quantum Algorithm.md": "quantum computing research.",
        "Fact - Neural Networks.md": "deep learning architectures.",
        "Fact - Robot Control.md": "robotics feedback control loops.",
        "Fact - Game Theory.md": "algorithmic game theory mechanism design."
    }
    
    for filename, content in papers.items():
        with open(wiki_dir / filename, 'w', encoding='utf-8') as f:
            f.write(f"---\ntags:\n  - fact_note\n---\n# {filename.replace('.md','')}\n{content}\n")
            
    # Verify personas are dynamically activated
    logger = MagicMock()
    personas = get_dynamic_personas([str(wiki_dir / f) for f in papers.keys()], logger)
    assert "Quantum Physicist" in personas
    assert "Deep Learning Architect" in personas
    assert "Robotics & Control Engineer" in personas
    assert "Algorithmic Game Theorist" in personas
    
    # Mock Gemini debate output
    mock_model = MagicMock()
    mock_model_class.return_value = mock_model
    
    mock_response = MagicMock()
    mock_response.text = """# Synthesized Debate: Automated Control Dynamics
The core connection is the application of agent decision systems.
Meta-Analyst: Analyzing all inputs.
[[Fact - Quantum Algorithm]]
[[Fact - Neural Networks]]
[[Fact - Robot Control]]
[[Fact - Game Theory]]
"""
    mock_model.generate_content.return_value = mock_response
    
    # Run the debate script
    code = run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir)
    ])
    
    assert code == 0
    # Verify Relationship and Concept Hubs are generated and linked
    rel_files = os.listdir(synthesis_dir)
    assert len(rel_files) == 1
    assert rel_files[0].startswith("Relationship - ")
    
    concept_files = os.listdir(concepts_dir)
    assert len(concept_files) == 1
    assert concept_files[0].startswith("Concept - ")
    
    # Verify bidirectional links injected in source fact notes
    with open(wiki_dir / "Fact - Quantum Algorithm.md", 'r', encoding='utf-8') as f:
        fact_content = f.read()
    assert "[[Concept - Agent Decision Systems]]" in fact_content
    assert "[[Relationship - Automated Control Dynamics]]" in fact_content
    
    # Verify Vault compliance sweep passes
    violations = check_vault(str(wiki_dir))
    assert not violations, f"Obsidian compliance violations found: {violations}"

# ---------------------------------------------------------
# Scenario 5: Obsidian Compliance Sweep
# ---------------------------------------------------------

def test_scenario5_compliance_sweep(tmp_path):
    # 1. Set up vault directories
    vault_dir = tmp_path / "mock_vault"
    os.makedirs(vault_dir)
    
    # Create ignored directories
    ignored_subdirs = [".obsidian", ".git", ".pytest_cache", "__pycache__", "raw", "tests", ".agents"]
    for subdir in ignored_subdirs:
        path = vault_dir / subdir
        os.makedirs(path)
        # Create files with heavy violations inside ignored subdirectories - should not be checked
        with open(path / "should_be_ignored.md", "w", encoding="utf-8") as f:
            f.write("tags: not_a_list\n#NoSpaceHeading\n[[unclosed_link")

    # Create active check directories
    os.makedirs(vault_dir / "wiki")
    os.makedirs(vault_dir / "wiki" / "concepts")
    os.makedirs(vault_dir / "wiki" / "synthesis")

    # 2. Populate compliant files (should produce 0 violations)
    valid_fact = vault_dir / "wiki" / "Fact - Valid.md"
    with open(valid_fact, "w", encoding="utf-8") as f:
        f.write("---\ntags:\n  - fact_note\n  - education\n---\n# Valid Fact Title\nThis is [[Concept - Learning]].\n")

    valid_concept = vault_dir / "wiki" / "concepts" / "Concept - Learning.md"
    with open(valid_concept, "w", encoding="utf-8") as f:
        f.write("---\ntags:\n  - concept_node\n---\n# Concept: Learning\nRefer to [[Fact - Valid]].\n")

    # 3. Populate ignored files in active check directories (should produce 0 violations)
    ignored_files = ["README.md", "Dashboard.md", "TEST_INFRA.md", "SYSTEM_HARNESS.md"]
    for filename in ignored_files:
        path = vault_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write("tags: invalid_format\n#NoSpaceHeading\n")

    # 4. Populate violating files (must produce specific violations)
    
    # File A: YAML boundary violation
    file_yaml_boundary = vault_dir / "wiki" / "Fact - BadYAML.md"
    with open(file_yaml_boundary, "w", encoding="utf-8") as f:
        f.write("tags:\n  - tag_name\n---\n# Title\n")

    # File B: Non-list tags field
    file_bad_tags = vault_dir / "wiki" / "Fact - BadTags.md"
    with open(file_bad_tags, "w", encoding="utf-8") as f:
        f.write("---\ntags: scalar_tag_instead_of_list\n---\n# Title\n")

    # File C: Block tags list items not starting with '-'
    file_bad_tag_block = vault_dir / "wiki" / "Fact - BadTagBlock.md"
    with open(file_bad_tag_block, "w", encoding="utf-8") as f:
        f.write("---\ntags:\n  [tag1, tag2]\n---\n# Title\n")

    # File D: Mismatched / Nested wikilinks
    file_mismatched_links = vault_dir / "wiki" / "Fact - BadLinks.md"
    with open(file_mismatched_links, "w", encoding="utf-8") as f:
        f.write("---\ntags:\n  - fact_note\n---\n# Title\n[[Fact - A [[Fact - B]]]] and [[unclosed_bracket\n")

    # File E: Heading missing space
    file_bad_heading = vault_dir / "wiki" / "synthesis" / "Relationship - BadHeading.md"
    with open(file_bad_heading, "w", encoding="utf-8") as f:
        f.write("---\ntags:\n  - relationship_node\n---\n#NoSpaceHeading\n> #NoSpaceInBlockquote\n- #NoSpaceInList\n")

    # File F: Valid complex bypasses (valid headings inside code/comments, brackets in code)
    file_valid_bypasses = vault_dir / "wiki" / "Fact - Bypasses.md"
    with open(file_valid_bypasses, "w", encoding="utf-8") as f:
        f.write("""---
tags:
  - fact_note
---
# Valid Title

```python
#NoSpaceComment in code block is fine
grid = [[1], [2]] # Double brackets in code block are fine
```

<!-- #NoSpaceHeading in HTML comment is fine -->
%% [[Mismatched bracket in Obsidian comment %%
""")

    # 5. Run compliance check
    violations_raw = check_vault(str(vault_dir))
    # Normalize keys to use forward slashes for cross-platform matching
    violations = {k.replace('\\', '/'): v for k, v in violations_raw.items()}

    # 6. Perform assertions
    # Verify compliant files and ignored files/folders have NO violations
    assert "wiki/Fact - Valid.md" not in violations
    assert "wiki/concepts/Concept - Learning.md" not in violations
    assert "wiki/Fact - Bypasses.md" not in violations
    for filename in ignored_files:
        assert filename not in violations
    for subdir in ignored_subdirs:
        assert not any(k.startswith(subdir) for k in violations.keys())

    # Verify expected violations are caught
    
    # File A: YAML boundary
    rel_yaml = "wiki/Fact - BadYAML.md"
    assert rel_yaml in violations
    assert any("YAML boundary" in v for v in violations[rel_yaml])

    # File B: Non-list tags
    rel_tags = "wiki/Fact - BadTags.md"
    assert rel_tags in violations
    assert any("tags" in v.lower() and "list" in v.lower() for v in violations[rel_tags])

    # File C: Block tags items not starting with '-'
    rel_tag_block = "wiki/Fact - BadTagBlock.md"
    assert rel_tag_block in violations
    assert any("YAML 'tags' list items do not start with '-'" in v for v in violations[rel_tag_block])

    # File D: Mismatched / Nested wikilinks
    rel_links = "wiki/Fact - BadLinks.md"
    assert rel_links in violations
    assert any("Nested wikilinks" in v for v in violations[rel_links])
    assert any("orphaned open brackets" in v for v in violations[rel_links])

    # File E: Heading format
    rel_heading = "wiki/synthesis/Relationship - BadHeading.md"
    assert rel_heading in violations
    bad_headings = [v for v in violations[rel_heading] if "Heading format is incorrect" in v]
    assert len(bad_headings) == 3
