import os
import tempfile
import xml.etree.ElementTree as ET
import urllib.request
import pytest

def test_urlopen_mock():
    """Verify that urllib.request.urlopen mock intercepts arXiv queries and returns the expected XML."""
    url = "http://export.arxiv.org/api/query?search_query=all:AI"
    response = urllib.request.urlopen(url)
    data = response.read()
    
    # Parse XML feed
    root = ET.fromstring(data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('atom:entry', ns)
    
    assert len(entries) == 2
    
    titles = [entry.find('atom:title', ns).text.strip() for entry in entries]
    assert "AI Agents in Education" in titles
    assert "Reinforcement Learning for Autonomous Cars" in titles
    
    # Verify PDF URLs exist
    pdf_urls = []
    for entry in entries:
        for link in entry.findall('atom:link', ns):
            if link.attrib.get('title') == 'pdf':
                pdf_urls.append(link.attrib.get('href'))
                
    assert len(pdf_urls) == 2
    assert any("2401.00001" in url for url in pdf_urls)
    assert any("2401.00002" in url for url in pdf_urls)

def test_urlretrieve_mock():
    """Verify that urllib.request.urlretrieve mock writes dummy PDF content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_file = os.path.join(tmpdir, "test_paper.pdf")
        url = "http://export.arxiv.org/pdf/2401.00001.pdf"
        
        filename, headers = urllib.request.urlretrieve(url, filename=dest_file)
        
        assert filename == dest_file
        assert os.path.exists(dest_file)
        with open(dest_file, 'rb') as f:
            content = f.read()
        assert b"%PDF-1.4" in content
        assert b"%%EOF" in content

def test_gemini_generate_content_fact():
    """Verify that google.generativeai mock returns a valid Fact note based on the prompt."""
    import google.generativeai as genai
    
    model = genai.GenerativeModel("gemini-pro")
    
    # Test Fact extraction prompt
    response = model.generate_content("Extract facts for: Reinforcement Learning for Autonomous Cars")
    text = response.text
    
    # Verify Obsidian compliance
    lines = text.strip().split('\n')
    assert lines[0] == '---'
    
    # Find closing frontmatter boundary
    closing_idx = -1
    for idx, line in enumerate(lines[1:], start=1):
        if line == '---':
            closing_idx = idx
            break
    assert closing_idx != -1
    
    # Verify frontmatter tags are a list
    frontmatter = lines[1:closing_idx]
    has_tags = False
    for idx, line in enumerate(frontmatter):
        if line.startswith('tags:'):
            has_tags = True
            # Verify the next lines are list items starting with '-'
            assert frontmatter[idx+1].strip().startswith('-')
            break
    assert has_tags
    
    # Verify content keywords
    assert "Fact Note: Reinforcement Learning for Autonomous Cars" in text
    assert "Hypothesis 1" in text

def test_gemini_generate_content_debate():
    """Verify that google.generativeai mock returns a valid Debate transcript based on the prompt."""
    import google.generativeai as genai
    
    model = genai.GenerativeModel("gemini-pro")
    
    # Test Debate synthesis prompt
    response = model.generate_content("Run a colosseum debate on AI agents in education vs autonomous cars")
    text = response.text
    
    # Verify Obsidian compliance
    lines = text.strip().split('\n')
    assert lines[0] == '---'
    
    # Find closing frontmatter boundary
    closing_idx = -1
    for idx, line in enumerate(lines[1:], start=1):
        if line == '---':
            closing_idx = idx
            break
    assert closing_idx != -1
    
    # Verify frontmatter tags and WikiLinks
    frontmatter = lines[1:closing_idx]
    has_source_papers = False
    for idx, line in enumerate(frontmatter):
        if line.startswith('source_papers:'):
            has_source_papers = True
            # Verify they are WikiLinks
            assert "[[" in frontmatter[idx+1]
            assert "]]" in frontmatter[idx+1]
            break
    assert has_source_papers
    
    # Verify content keywords
    assert "Synthesized Debate:" in text
    assert "Meta-Analyst:" in text
