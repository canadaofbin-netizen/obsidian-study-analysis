import os
import sys
import pytest
import logging
from unittest.mock import patch, MagicMock

# Ensure scripts directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from run_debate import main as run_debate, parse_args, get_dynamic_personas
from compliance_checker import check_file

def test_parse_args_defaults():
    """Verify that command line arguments parse correctly with default values."""
    args = parse_args([])
    assert args.wiki_dir is None
    assert args.synthesis_dir is None
    assert args.concepts_dir is None
    assert args.prompt_path is None
    assert args.model_name == "gemini-pro"
    assert args.verbose is False
    assert args.log_path is None

def test_parse_args_overrides():
    """Verify CLI overrides are captured properly."""
    args = parse_args([
        '--wiki-dir', 'my_wiki',
        '--synthesis-dir', 'my_synthesis',
        '--concepts-dir', 'my_concepts',
        '--prompt-path', 'my_prompt.md',
        '--model-name', 'gemini-1.5-flash',
        '--verbose',
        '--log-path', 'my_log.log'
    ])
    assert args.wiki_dir == 'my_wiki'
    assert args.synthesis_dir == 'my_synthesis'
    assert args.concepts_dir == 'my_concepts'
    assert args.prompt_path == 'my_prompt.md'
    assert args.model_name == 'gemini-1.5-flash'
    assert args.verbose is True
    assert args.log_path == 'my_log.log'

def test_run_debate_no_fact_notes(tmp_path, caplog):
    """Verify that run_debate exits gracefully when no Fact notes are present."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    os.makedirs(wiki_dir)
    
    with caplog.at_level(logging.WARNING, logger="DebateEngine"):
        code = run_debate([
            '--wiki-dir', str(wiki_dir),
            '--synthesis-dir', str(synthesis_dir),
            '--concepts-dir', str(concepts_dir)
        ])
    
    assert code == 0
    assert "No Fact notes found" in caplog.text
    # Ensure no relationship or concept files were created inside the directories
    assert len(os.listdir(synthesis_dir)) == 0
    assert len(os.listdir(concepts_dir)) == 0

@patch('google.generativeai.GenerativeModel')
def test_run_debate_successful_run(mock_model_class, tmp_path):
    """Verify a complete successful run of the debate script with mock Gemini API and check Obsidian compliance."""
    wiki_dir = tmp_path / "wiki"
    synthesis_dir = wiki_dir / "synthesis"
    concepts_dir = wiki_dir / "concepts"
    prompt_path = tmp_path / "orchestrator_prompt.md"
    
    os.makedirs(wiki_dir)
    
    # Create fact notes
    fact1_content = "---\ntags:\n  - fact_note\n---\n# Fact Note: Paper One\n## Related Concepts\n"
    fact2_content = "---\ntags:\n  - fact_note\n---\n# Fact Note: Paper Two\n"
    
    with open(wiki_dir / "Fact - Paper One.md", "w", encoding="utf-8") as f:
        f.write(fact1_content)
    with open(wiki_dir / "Fact - Paper Two.md", "w", encoding="utf-8") as f:
        f.write(fact2_content)
        
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write("System prompt template instructions")
        
    # Setup mock Gemini behavior
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance
    mock_response = MagicMock()
    mock_response.text = """# Synthesized Debate: Semantic Search Intersections
The core connection is the application of semantic search.
[[Fact - Paper One]]
[[Fact - Paper Two]]
"""
    mock_instance.generate_content.return_value = mock_response
    
    # Run the debate script
    code = run_debate([
        '--wiki-dir', str(wiki_dir),
        '--synthesis-dir', str(synthesis_dir),
        '--concepts-dir', str(concepts_dir),
        '--prompt-path', str(prompt_path),
        '--model-name', 'gemini-pro',
        '--verbose'
    ])
    
    assert code == 0
    
    # 1. Verify model instantiation passed system instruction properly
    mock_model_class.assert_called_once_with(
        model_name='gemini-pro',
        system_instruction="System prompt template instructions"
    )
    
    # 2. Verify files created
    assert os.path.exists(synthesis_dir)
    assert os.path.exists(concepts_dir)
    
    # Verify Relationship note content and filename
    rel_files = os.listdir(synthesis_dir)
    assert len(rel_files) == 1
    assert rel_files[0].startswith("Relationship - ")
    rel_file_path = synthesis_dir / rel_files[0]
    with open(rel_file_path, "r", encoding="utf-8") as f:
        rel_content = f.read()
    assert "Synthesized Debate: Semantic Search Intersections" in rel_content
    # Check that frontmatter was automatically injected
    assert rel_content.startswith("---")
    assert "relationship_note" in rel_content
    assert "concept_hub: \"[[Concept - Semantic Search]]\"" in rel_content
    
    # Verify Concept Hub note content and filename
    concept_files = os.listdir(concepts_dir)
    assert len(concept_files) == 1
    assert concept_files[0].startswith("Concept - ")
    concept_file_path = concepts_dir / concept_files[0]
    with open(concept_file_path, "r", encoding="utf-8") as f:
        concept_content = f.read()
    assert "Concept Hub: Semantic Search" in concept_content
    # Check concept hub has backlink to relationship note
    rel_title = os.path.splitext(rel_files[0])[0]
    assert f"[[{rel_title}]]" in concept_content
    
    # 3. Verify bidirectional link injection in Fact notes
    with open(wiki_dir / "Fact - Paper One.md", "r", encoding="utf-8") as f:
        updated_fact1 = f.read()
    assert "[[Concept - Semantic Search]]" in updated_fact1
    assert f"[[{rel_title}]]" in updated_fact1
    
    with open(wiki_dir / "Fact - Paper Two.md", "r", encoding="utf-8") as f:
        updated_fact2 = f.read()
    assert "[[Concept - Semantic Search]]" in updated_fact2
    assert f"[[{rel_title}]]" in updated_fact2

    # 4. Verify Obsidian compliance via compliance_checker
    violations = check_file(str(rel_file_path))
    assert len(violations) == 0, f"Relationship note violations: {violations}"
    
    violations = check_file(str(concept_file_path))
    assert len(violations) == 0, f"Concept hub violations: {violations}"

@patch('google.generativeai.GenerativeModel')
def test_run_debate_missing_prompt_path(mock_model_class, tmp_path, caplog):
    """Verify how run_debate handles a missing prompt template path."""
    wiki_dir = tmp_path / "wiki"
    os.makedirs(wiki_dir)
    with open(wiki_dir / "Fact - Paper One.md", "w", encoding="utf-8") as f:
        f.write("# Fact Note: Paper One")
    with open(wiki_dir / "Fact - Paper Two.md", "w", encoding="utf-8") as f:
        f.write("# Fact Note: Paper Two")
        
    invalid_prompt_path = tmp_path / "nonexistent.md"
    
    # Mock LLM response
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance
    mock_response = MagicMock()
    mock_response.text = "Debate output content"
    mock_instance.generate_content.return_value = mock_response
    
    with caplog.at_level(logging.WARNING, logger="DebateEngine"):
        run_debate([
            '--wiki-dir', str(wiki_dir),
            '--synthesis-dir', str(tmp_path / "synthesis"),
            '--concepts-dir', str(tmp_path / "concepts"),
            '--prompt-path', str(invalid_prompt_path)
        ])
        
    assert "Prompt template file not found" in caplog.text
    # Should still proceed without system_instruction
    mock_model_class.assert_called_once_with(
        model_name='gemini-pro',
        system_instruction=None
    )

@patch('google.generativeai.GenerativeModel')
def test_run_debate_api_error_handling(mock_model_class, tmp_path, caplog):
    """Verify that Gemini API errors are caught and logged, and the script exits with status 1."""
    wiki_dir = tmp_path / "wiki"
    os.makedirs(wiki_dir)
    with open(wiki_dir / "Fact - Paper One.md", "w", encoding="utf-8") as f:
        f.write("# Fact Note: Paper One")
    with open(wiki_dir / "Fact - Paper Two.md", "w", encoding="utf-8") as f:
        f.write("# Fact Note: Paper Two")
        
    mock_instance = MagicMock()
    mock_model_class.return_value = mock_instance
    # Simulate API exception
    mock_instance.generate_content.side_effect = Exception("API Quota exceeded")
    
    with pytest.raises(SystemExit) as excinfo:
        run_debate([
            '--wiki-dir', str(wiki_dir),
            '--synthesis-dir', str(tmp_path / "synthesis"),
            '--concepts-dir', str(tmp_path / "concepts")
        ])
        
    assert excinfo.value.code == 1

def test_get_dynamic_personas(tmp_path):
    """Verify that specialized personas are activated dynamically when keywords are present in Fact notes."""
    logger = logging.getLogger("TestLogger")
    
    # Case 1: No keywords
    fact1 = tmp_path / "Fact - Normal.md"
    with open(fact1, "w", encoding="utf-8") as f:
        f.write("# Fact Note: Normal paper\nDoes not contain any keywords.")
    personas = get_dynamic_personas([str(fact1)], logger)
    assert "Quantum Physicist" not in personas
    assert "Deep Learning Architect" not in personas
    
    # Case 2: Quantum keyword
    fact2 = tmp_path / "Fact - Quantum.md"
    with open(fact2, "w", encoding="utf-8") as f:
        f.write("# Fact Note: Quantum Mechanics\nThis paper discusses quantum computing and algorithms.")
    personas = get_dynamic_personas([str(fact2)], logger)
    assert "Quantum Physicist" in personas
    assert personas["Quantum Physicist"] == "Analyzes physical and quantum mechanical concepts, quantum computing, and state superposition."
    assert "Deep Learning Architect" not in personas
    
    # Case 3: Deep learning keywords
    fact3 = tmp_path / "Fact - Transformer.md"
    with open(fact3, "w", encoding="utf-8") as f:
        f.write("# Fact Note: Attention is All You Need\nThis paper focuses on transformer neural networks and deep learning architectures.")
    personas = get_dynamic_personas([str(fact3)], logger)
    assert "Deep Learning Architect" in personas
    assert "Quantum Physicist" not in personas
