import os
import sys
import types
import pytest
import re
import time
from unittest.mock import patch, MagicMock

# Robust sys.path configuration to ensure consistent imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_scripts_dir = os.path.join(_project_root, "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# Define helper for LLM response generation based on prompt content
def generate_mock_llm_response(prompt_text):
    prompt_lower = prompt_text.lower()
    
    # Check if this is a debate or synthesis request
    if any(keyword in prompt_lower for keyword in ["debate", "synthesis", "relationship", "colosseum"]):
        # Extract fact notes from the prompt
        papers = re.findall(r'\[\[(Fact - [^\]]+)\]\]', prompt_text)
        if not papers:
            # Try to match Fact - ... in the prompt
            papers = re.findall(r'(Fact - [a-zA-Z0-9_\- \t\.\(\)\&]+)', prompt_text)
        
        # Clean them
        cleaned_papers = []
        for p in papers:
            p_clean = p.replace(".md", "").strip()
            p_clean = re.sub(r'[\.,\*\?\!]+$', '', p_clean)
            if p_clean and p_clean not in cleaned_papers:
                cleaned_papers.append(p_clean)
                
        if not cleaned_papers:
            cleaned_papers = ["Fact - AI Agents in Education", "Fact - Reinforcement Learning for Autonomous Cars"]
            
        # Ensure they are in [[Fact - Title]] format
        formatted_papers = [f"[[{p}]]" if not p.startswith("[[") else p for p in cleaned_papers]
        
        # Determine theme dynamically based on first paper title
        first_paper_clean = cleaned_papers[0].replace("Fact - ", "")
        words = [w for w in re.split(r'[^a-zA-Z0-9]', first_paper_clean) if w]
        relationship_theme = " ".join(words[:4]) if words else "AI Agent Advancements"
        
        source_papers_yaml = "\n".join(f"  - \"{p}\"" for p in formatted_papers)
        
        return f"""---
tags:
  - relationship_node
  - debate_transcript
rel_type: extension
source_papers:
{source_papers_yaml}
---

# Synthesized Debate: {relationship_theme}

## The Internal Debate

**Meta-Analyst:** Welcome to the Colosseum Debate. Today we analyze the integration of research findings on AI agents.
We have papers: {", ".join(formatted_papers)}.

**Devil's Advocate:** I argue that these papers share no common ground. They focus on entirely different areas.

**Causal Inference Specialist:** Both systems rely on decision-making models. Both map state spaces to action policies.

**Meta-Analyst:** Let's synthesize this. The core connection is the application of sequential decision-making under uncertainty.
"""
    else:
        # Fact extraction request
        # E.g. prompt: "Extract facts for: AI Agents in Education.pdf"
        match = re.search(r'Extract facts for:\s*([^\n]+)', prompt_text)
        if match:
            filename = match.group(1).strip()
            title = os.path.splitext(filename)[0]
        else:
            title = "AI Agents in Education"
            if "autonomous" in prompt_lower or "car" in prompt_lower:
                title = "Reinforcement Learning for Autonomous Cars"
        
        clean_tag = title.lower().replace(" ", "_")
        clean_tag = "".join(c for c in clean_tag if c.isalnum() or c == "_")
        
        return f"""---
tags:
  - fact_note
  - {clean_tag}
---

# Fact Note: {title}
**Author:** Scholar Bot

## 1. Explicit Hypotheses & Core Arguments
* **Hypothesis 1:** AI agents can optimize state representations to improve performance.

## 2. Methodologies
* Simulation-based evaluation and statistical analysis.

## 3. Empirical & Technical Findings
* Implementation achieved 95% efficiency compared to baseline models.
"""

# Setup mock class for google.generativeai.GenerativeModel
class MockGenerativeModel:
    def __init__(self, model_name, *args, **kwargs):
        self.model_name = model_name

    def generate_content(self, contents, *args, **kwargs):
        prompt_text = str(contents)
        response_text = generate_mock_llm_response(prompt_text)
        mock_resp = MagicMock()
        mock_resp.text = response_text
        return mock_resp

def mock_configure(*args, **kwargs):
    pass

@pytest.fixture(autouse=True)
def mock_time_sleep():
    """Mock time.sleep globally to avoid slowing down tests, except for small sleeps (< 0.5s)."""
    original_sleep = time.sleep
    def dummy_sleep(seconds):
        if seconds < 0.5:
            original_sleep(seconds)
    with patch('time.sleep', side_effect=dummy_sleep) as p:
        yield p

# Insert the mock module into sys.modules to handle environment without google-generativeai installed
if 'google' not in sys.modules:
    google_mod = types.ModuleType('google')
    sys.modules['google'] = google_mod

if 'google.generativeai' not in sys.modules:
    genai_mod = types.ModuleType('google.generativeai')
    genai_mod.GenerativeModel = MockGenerativeModel
    genai_mod.configure = mock_configure
    sys.modules['google.generativeai'] = genai_mod
    sys.modules['google'].generativeai = genai_mod

@pytest.fixture(autouse=True)
def mock_urllib_urlopen():
    """Mock urllib.request.urlopen to intercept arXiv API queries."""
    from urllib.request import Request
    
    mock_xml = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>AI Agents in Education</title>
    <link title="pdf" href="http://export.arxiv.org/pdf/2401.00001" />
  </entry>
  <entry>
    <title>Reinforcement Learning for Autonomous Cars</title>
    <link title="pdf" href="http://export.arxiv.org/pdf/2401.00002" />
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

    def mock_urlopen(url, *args, **kwargs):
        url_str = ""
        if isinstance(url, str):
            url_str = url
        elif isinstance(url, Request):
            url_str = url.full_url
        elif hasattr(url, 'full_url'):
            url_str = url.full_url
        elif hasattr(url, 'get_full_url'):
            url_str = url.get_full_url()
        
        if "arxiv.org" in url_str:
            return MockHTTPResponse(mock_xml.encode('utf-8'))
        return MockHTTPResponse(b"<feed></feed>")

    with patch('urllib.request.urlopen', side_effect=mock_urlopen) as p:
        yield p

@pytest.fixture(autouse=True)
def mock_urllib_urlretrieve():
    """Mock urllib.request.urlretrieve to write a dummy PDF file."""
    def mock_urlretrieve(url, filename=None, *args, **kwargs):
        if filename:
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            dummy_pdf = b"%PDF-1.4\n%dummy pdf content\n%%EOF"
            with open(filename, 'wb') as f:
                f.write(dummy_pdf)
        return (filename, MagicMock())

    with patch('urllib.request.urlretrieve', side_effect=mock_urlretrieve) as p:
        yield p

@pytest.fixture(autouse=True)
def mock_gemini_generate_content():
    """Fixture to ensure the mock GenerativeModel is active and yields its generate_content patch."""
    import google.generativeai as genai
    
    with patch.object(genai.GenerativeModel, 'generate_content', autospec=True) as p:
        def dynamic_generate(self, contents, *args, **kwargs):
            prompt_text = str(contents)
            response_text = generate_mock_llm_response(prompt_text)
            mock_resp = MagicMock()
            mock_resp.text = response_text
            return mock_resp
            
        p.side_effect = dynamic_generate
        yield p

@pytest.fixture(autouse=True)
def mock_pdf_reader_globally():
    """Globally mock pypdf.PdfReader for all tests to return long dummy text containing the filename keywords."""
    import pypdf
    
    class DynamicPdfReader:
        def __init__(self, stream, *args, **kwargs):
            self.stream = stream
            self.is_encrypted = False
            
            # Resolve title from path/stream
            filename = ""
            if isinstance(stream, str):
                filename = os.path.basename(stream)
            elif hasattr(stream, 'name'):
                filename = os.path.basename(stream.name)
                
            title = "AI Agents in Education"
            if "autonomous" in filename.lower() or "car" in filename.lower():
                title = "Reinforcement Learning for Autonomous Cars"
            elif "test" in filename.lower():
                title = "Test PDF Document"
                
            mock_page = MagicMock()
            mock_page.extract_text.return_value = (
                f"This is a dummy extracted PDF text for the study titled '{title}' that is long enough to pass "
                f"the length check of 100 characters. It contains hypotheses, methodology, and empirical findings "
                f"for the research paper on {title}. Let's make sure it contains enough characters to satisfy the guards."
            )
            self.pages = [mock_page]
            
        def decrypt(self, password):
            return 1

    with patch('pypdf.PdfReader', side_effect=DynamicPdfReader) as p:
        yield p
