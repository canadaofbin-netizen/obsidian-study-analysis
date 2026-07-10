import argparse
import os
import re
import logging
import sys
import google.generativeai as genai

# Default personas representing core agents
PERSONAS = {
    "Meta-Analyst": "Focuses on synthesis, structure, and identifying common ground.",
    "Devil's Advocate": "Challenges assumptions, questions evidence quality, raises alternative explanations.",
    "Causal Inference Specialist": "Focuses on underlying causal models, mechanism design, and validity of empirical inferences."
}

def load_personas():
    return PERSONAS

def get_dynamic_personas(fact_notes_paths, logger):
    """Scan fact note contents and dynamically activate specialized personas based on keywords."""
    personas = PERSONAS.copy()
    
    # Keyword to specialized persona mapping
    specialized_mappings = {
        r"(?i)\bquantums?\b": ("Quantum Physicist", "Analyzes physical and quantum mechanical concepts, quantum computing, and state superposition."),
        r"(?i)\bneural networks?\b|\bdeep learnings?\b|\btransformers?\b": ("Deep Learning Architect", "Focuses on neural network layers, transformer attention mechanisms, and training dynamics."),
        r"(?i)\brobot(?:s|ics)?\b|\bcontrol theor(?:y|ies)\b|\bautonomous\b": ("Robotics & Control Engineer", "Focuses on feedback loops, sensor integration, and physical system dynamics."),
        r"(?i)\beconomics?\b|\bgame theor(?:y|ies)\b|\bincentives?\b": ("Algorithmic Game Theorist", "Focuses on mechanism design, utility functions, and equilibrium states."),
        r"(?i)\bagent-based\b|\bmulti-agents?\b": ("Multi-Agent Coordinator", "Focuses on communication protocols, emergent behaviors, and collaborative tasks.")
    }
    
    for path in fact_notes_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            for pattern, (name, desc) in specialized_mappings.items():
                if name not in personas and re.search(pattern, content):
                    personas[name] = desc
                    logger.info(f"Dynamically activated specialized persona: {name}")
        except Exception as e:
            logger.warning(f"Failed to read {path} for dynamic persona analysis: {e}")
            
    return personas

def setup_logging(verbose=False, log_path=None):
    """Configure python logging with console and file output."""
    logger = logging.getLogger("DebateEngine")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Clear previous handlers except pytest caplog handlers
    logger.handlers = [h for h in logger.handlers if h.__class__.__name__ == 'LogCaptureHandler']
    logger.propagate = True  # Propagate logs to root logger for pytest caplog capture
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            fh = logging.FileHandler(log_path, encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            logger.error(f"Failed to create log file handler: {e}")
            
    return logger

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run Academic Colosseum debate between research facts")
    parser.add_argument('--wiki-dir', type=str, default=None, help="Directory containing Fact notes")
    parser.add_argument('--synthesis-dir', type=str, default=None, help="Directory to save Relationship notes")
    parser.add_argument('--concepts-dir', type=str, default=None, help="Directory to save Concept Hubs")
    parser.add_argument('--prompt-path', type=str, default=None, help="Path to prompt template")
    parser.add_argument('--model-name', type=str, default="gemini-pro", help="Gemini model to run debate")
    parser.add_argument('--verbose', action='store_true', help="Enable debug/verbose log level")
    parser.add_argument('--log-path', type=str, default=None, help="Log execution info to this file")
    return parser.parse_args(args)

def clean_llm_markdown(text):
    """Strip out markdown code block wraps if the LLM wrapped the output."""
    text = text.strip()
    pattern = r'^```(?:markdown)?\s*\n(.*?)\n```$'
    match = re.match(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text

def inject_link_safely(content, header, link):
    """Only insert link if the header is found outside code blocks.
    Otherwise, append the new section/link to the end of the file.
    """
    if link in content:
        return content, False

    lines = content.splitlines()
    in_code_block = False
    header_line_idx = -1
    
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
        elif not in_code_block and stripped == header:
            header_line_idx = idx
            
    if header_line_idx != -1:
        lines.insert(header_line_idx + 1, f"* {link}")
        return "\n".join(lines), True
    else:
        new_content = content.rstrip() + f"\n\n{header}\n* {link}\n"
        return new_content, True

def main(args=None):
    parsed_args = parse_args(args)
    logger = setup_logging(parsed_args.verbose, parsed_args.log_path)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)
    
    wiki_dir = parsed_args.wiki_dir or os.path.join(workspace_root, "wiki")
    synthesis_dir = parsed_args.synthesis_dir or os.path.join(wiki_dir, "synthesis")
    concepts_dir = parsed_args.concepts_dir or os.path.join(wiki_dir, "concepts")
    prompt_path = parsed_args.prompt_path or os.path.join(workspace_root, "prompts", "2_academic_orchestrator.md")
    
    # Ensure directories exist
    for d in [wiki_dir, synthesis_dir, concepts_dir]:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
                logger.debug(f"Created directory: {d}")
            except Exception as e:
                logger.error(f"Failed to create directory {d}: {e}")
                sys.exit(1)
                
    # Scan for Fact notes
    fact_notes = []
    if os.path.exists(wiki_dir):
        for f in os.listdir(wiki_dir):
            if f.startswith("Fact - ") and f.endswith(".md"):
                fact_notes.append(f)
                
    if len(fact_notes) < 2:
        if len(fact_notes) == 0:
            logger.warning(f"No Fact notes found in directory: {wiki_dir}. Aborting debate execution.")
        else:
            logger.warning(f"Fewer than 2 Fact notes found in directory: {wiki_dir}. Aborting debate execution.")
        return 0
        
    logger.info(f"Found {len(fact_notes)} Fact notes.")
    
    # Load system prompt
    system_prompt = None
    if os.path.exists(prompt_path):
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
            logger.info(f"Loaded system prompt from {prompt_path}")
        except Exception as e:
            logger.error(f"Error reading prompt template from {prompt_path}: {e}")
    else:
        logger.warning(f"Prompt template file not found at {prompt_path}. Proceeding without system instruction.")
        
    # Configure Gemini and instantiate model with system instruction
    api_key = os.environ.get("GEMINI_API_KEY", "dummy-key")
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(
            model_name=parsed_args.model_name,
            system_instruction=system_prompt if system_prompt else None
        )
    except Exception as e:
        logger.error(f"Failed to initialize GenerativeModel with name {parsed_args.model_name}: {e}")
        sys.exit(1)
        
    # Dynamic Personas Activation
    fact_notes_paths = [os.path.join(wiki_dir, f) for f in fact_notes]
    personas = get_dynamic_personas(fact_notes_paths, logger)
    logger.info(f"Active personas for debate: {list(personas.keys())}")
    
    # Build user prompt
    prompt = f"Run a colosseum debate on: " + ", ".join(fact_notes)
    prompt += "\nPersonas involved:\n" + "\n".join(f"- {name}: {desc}" for name, desc in personas.items())
    
    logger.info("Calling Gemini API to simulate debate...")
    try:
        response = model.generate_content(prompt)
        raw_transcript = response.text
        if not raw_transcript:
            raise ValueError("API returned an empty transcript.")
        debate_transcript = clean_llm_markdown(raw_transcript)
    except Exception as e:
        logger.error(f"Gemini API execution failed: {e}")
        sys.exit(1)
        
    # Derive relationship theme dynamically
    relationship_theme = None
    theme_match = re.search(r'^#\s*(?:Synthesized Debate|Relationship):\s*([^\n]+)', debate_transcript, re.MULTILINE)
    if theme_match:
        relationship_theme = theme_match.group(1).strip()
    else:
        first_paper_clean = fact_notes[0].replace("Fact - ", "").replace(".md", "").strip()
        words = [w for w in re.split(r'[^a-zA-Z0-9]', first_paper_clean) if w]
        relationship_theme = " ".join(words[:4]) if words else "AI Agent Advancements"
        
    relationship_theme = "".join(c for c in relationship_theme if c.isalnum() or c in " _-").strip()
    if not relationship_theme:
        relationship_theme = "AI Agent Advancements"
    relationship_filename = f"Relationship - {relationship_theme}.md"
    relationship_path = os.path.join(synthesis_dir, relationship_filename)
    
    # Extract source papers for Obsidian tags & frontmatter structure
    source_papers = re.findall(r'\[\[(Fact - [^\]]+)\]\]', debate_transcript)
    source_papers = [p.split('|')[0].strip() for p in source_papers]
    if not source_papers:
        source_papers = [os.path.splitext(f)[0] for f in fact_notes]
    
    # Deduplicate papers
    seen_papers = set()
    deduped_source_papers = []
    for p in source_papers:
        if p not in seen_papers:
            seen_papers.add(p)
            deduped_source_papers.append(p)
    source_papers = deduped_source_papers
    if len(source_papers) < 2:
        source_papers = [f.replace(".md", "") for f in fact_notes]
    
    # Derive Concept Title
    concept_title = None
    concept_match = re.search(r'(?:core connection is the application of|core connection is|intersection of)\s+([^\.\n]+?)(?:\.|\n|\Z)', debate_transcript, re.IGNORECASE)
    if concept_match:
        concept_title = concept_match.group(1).strip()
        concept_title = " ".join([w.capitalize() for w in concept_title.split()[:4]])
    if not concept_title:
        if len(fact_notes) > 1:
            second_paper_clean = fact_notes[1].replace("Fact - ", "").replace(".md", "").strip()
            words = [w for w in re.split(r'[^a-zA-Z0-9]', second_paper_clean) if w]
            concept_title = " ".join(words[:4]).title() if words else "Sequential Decision-Making"
        elif fact_notes:
            first_paper_clean = fact_notes[0].replace("Fact - ", "").replace(".md", "").strip()
            words = [w for w in re.split(r'[^a-zA-Z0-9]', first_paper_clean) if w]
            concept_title = " ".join(words[:4]).title() if words else "Sequential Decision-Making"
        else:
            concept_title = "Sequential Decision-Making"
            
    concept_title = "".join(c for c in concept_title if c.isalnum() or c in " _-").strip()
    if not concept_title:
        concept_title = "Sequential Decision-Making"
    concept_filename = f"Concept - {concept_title}.md"
    concept_path = os.path.join(concepts_dir, concept_filename)
    
    # Strip existing frontmatter if present to ensure clean wrapping
    content_to_write = debate_transcript
    if content_to_write.startswith("---"):
        parts = content_to_write.split("---", 2)
        if len(parts) >= 3:
            content_to_write = parts[2].strip()
            
    # Ensure Relationship Note has compliant YAML frontmatter
    source_papers_yaml = "\n".join(f"  - \"[[{p.replace('\"', '\\\"')}]]\"" for p in source_papers)
    frontmatter = f"""---
tags:
  - relationship_note
  - debate_transcript
source_papers:
{source_papers_yaml}
concept_hub: "[[Concept - {concept_title}]]"
---

"""
    debate_transcript = frontmatter + content_to_write
        
    try:
        with open(relationship_path, 'w', encoding='utf-8') as f:
            f.write(debate_transcript)
        logger.info(f"Generated Relationship note: {relationship_filename}")
    except Exception as e:
        logger.error(f"Failed to write Relationship note: {e}")
        sys.exit(1)
        
    # Generate Concept Hub note content with bidirectional link to Relationship note
    source_papers_links = "\n".join(f"- [[{p}]]" for p in source_papers)
    concept_content = f"""---
tags:
  - concept_hub
  - correlation_node
---

# Concept Hub: {concept_title}

This concept hub represents the theoretical intersection of {concept_title.lower()}.

## Supporting Fact Notes
{source_papers_links}

## Synthesis Debate
- [[Relationship - {relationship_theme}]]
"""
    try:
        with open(concept_path, 'w', encoding='utf-8') as f:
            f.write(concept_content)
        logger.info(f"Generated Concept Hub: {concept_filename}")
    except Exception as e:
        logger.error(f"Failed to write Concept Hub note: {e}")
        sys.exit(1)
        
    # Bidirectionally update Fact Notes to link to Concept Hub and Relationship note
    concept_link = f"[[Concept - {concept_title}]]"
    relationship_link = f"[[Relationship - {relationship_theme}]]"
    
    for p in source_papers:
        p_filename = f"{p}.md"
        p_path = os.path.join(wiki_dir, p_filename)
        if os.path.exists(p_path):
            try:
                with open(p_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                modified = False
                
                # Check / Inject Concept link
                content, changed_concept = inject_link_safely(content, "## Related Concepts", concept_link)
                if changed_concept:
                    modified = True
                    
                # Check / Inject Relationship link
                content, changed_relationship = inject_link_safely(content, "## Related Debates", relationship_link)
                if changed_relationship:
                    modified = True
                    
                if modified:
                    with open(p_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"Bidirectionally updated Fact Note: {p_filename}")
            except Exception as e:
                logger.error(f"Failed to update Fact Note {p_filename}: {e}")
                
    logger.info("Colosseum debate completed successfully.")
    return 0

if __name__ == "__main__":
    main()
