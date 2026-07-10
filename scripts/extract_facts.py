import argparse
import os
import sys
import json
import logging
import re
import pypdf
import google.generativeai as genai

# Setup module-level logger
logger = logging.getLogger("extract_facts")

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Extract facts from PDFs")
    parser.add_argument('--pdf-dir', type=str, default=None, help='Directory containing PDFs or path to a single PDF')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for fact notes')
    parser.add_argument('--state-path', type=str, default=None, help='Path to state file')
    parser.add_argument('--prompt-path', type=str, default='prompts/1_fact_extractor.md', help='Path to prompt template')
    parser.add_argument('--force', action='store_true', help='Reprocess even if already in state')
    parser.add_argument('--model-name', type=str, default='gemini-pro', help='Gemini model name')
    parser.add_argument('--log-path', type=str, default=None, help='Path to log file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose (DEBUG) logging')
    return parser.parse_args(args)

def configure_logging(log_path=None, verbose=False):
    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)
    # Clear any existing handlers to prevent duplication
    logger.handlers = []
    
    # Console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(log_level)
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)
    
    # File handler
    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        f_handler = logging.FileHandler(log_path, encoding='utf-8')
        f_handler.setLevel(log_level)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

def clean_and_format_markdown(text):
    # 1. Strip conversational code block wrappers if they wrap the content
    # Scan all code blocks.
    wrapper_match = None
    for cb in re.finditer(r'```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)```', text, re.DOTALL):
        content_inside = cb.group(1)
        stripped_inside = content_inside.strip()
        if stripped_inside.startswith('---') or ('tags:' in content_inside and '---' in content_inside):
            wrapper_match = cb
            break

    if wrapper_match:
        start_idx = wrapper_match.start() + wrapper_match.group(0).find(wrapper_match.group(1))
        end_idx = text.rfind('```')
        if end_idx > start_idx:
            text_stripped = text[start_idx:end_idx].strip()
        else:
            text_stripped = text[start_idx:].strip()
    else:
        text_stripped = text.strip()
        # Handle cases where code blocks are incomplete
        if text_stripped.startswith("```"):
            text_stripped = re.sub(r'^```(?:markdown|[a-zA-Z0-9_-]+)?\s*\n', '', text_stripped)
        if text_stripped.endswith("```"):
            text_stripped = re.sub(r'\n```$', '', text_stripped).strip()
                
    # 2. Standardize heading formatting: replace `#Heading` with `# Heading`
    # Ensure the markdown header cleaner regex does not modify `#` lines (comments, shebangs) that reside inside markdown code blocks (delimited by ``` or ~~~).
    in_code_block = False
    formatted_lines = []
    for line in text_stripped.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
            in_code_block = not in_code_block
            formatted_lines.append(line)
        else:
            if not in_code_block:
                line = re.sub(r'^([> \t-]*)(#{1,6})([^# \t\n])', r'\1\2 \3', line)
            formatted_lines.append(line)
    text_formatted = "\n".join(formatted_lines)
    
    # 3. Ensure the YAML frontmatter starts and ends with `---` at the top of the file
    # and the tags list format matches Obsidian requirements.
    lines = text_formatted.splitlines()
    
    if not lines:
        return ""
        
    # Check if lines start with '---'
    if lines[0].strip() != '---':
        lines = ['---', 'tags:', '  - fact_note', '---'] + lines
            
    # Find closing '---'
    closing_idx = -1
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == '---':
            is_valid = True
            if idx > 30:
                is_valid = False
            else:
                for fline in lines[1:idx]:
                    stripped = fline.strip()
                    if not stripped:
                        continue
                    if stripped.startswith('#'):
                        is_valid = False
                        break
                    if fline.startswith(' ') or fline.startswith('-'):
                        continue
                    if ':' in fline:
                        parts = fline.split(':', 1)
                        key = parts[0].strip()
                        if re.match(r'^[a-zA-Z0-9_-]+$', key):
                            continue
                    is_valid = False
                    break
            if is_valid:
                closing_idx = idx
                break
            
    if closing_idx == -1:
        # Closing boundary is missing. Let's find a reasonable boundary to insert it.
        boundary_idx = -1
        def is_body_bullet(line):
            if not line.startswith('- '):
                return False
            rest = line[2:]
            import string
            tag_punc = {'_', '-', '/'}
            has_space = any(c.isspace() for c in rest)
            has_punc = any(c in string.punctuation and c not in tag_punc for c in rest)
            return has_space or has_punc

        for idx, line in enumerate(lines[1:], start=1):
            if is_body_bullet(line):
                boundary_idx = idx
                break
            stripped = line.strip()
            # If it's a heading, it's the body
            if stripped.startswith('#'):
                boundary_idx = idx
                break
            # If line is not empty and doesn't belong to YAML structure (no colon, doesn't start with space or dash)
            if stripped and not line.startswith(' ') and not stripped.startswith('-') and ':' not in stripped:
                boundary_idx = idx
                break
        if boundary_idx != -1:
            lines.insert(boundary_idx, '---')
            closing_idx = boundary_idx
        else:
            lines.append('---')
            closing_idx = len(lines) - 1
            
    # Parse and rebuild YAML frontmatter
    import yaml
    yaml_block = "\n".join(lines[1:closing_idx])
    try:
        parsed_yaml = yaml.safe_load(yaml_block)
        if not isinstance(parsed_yaml, dict):
            parsed_yaml = {}
    except Exception:
        parsed_yaml = {}
        # Try a quick regex fallback for tags if yaml parsing failed
        tags_match = re.search(r'tags:\s*\[?(.*?)\]?$', yaml_block, re.MULTILINE)
        if tags_match:
            tags_str = tags_match.group(1)
            parsed_yaml['tags'] = [t.strip() for t in re.split(r'[,\s]+', tags_str) if t.strip() and t.strip() != '-']
            
    # Ensure tags is a list
    if 'tags' not in parsed_yaml:
        parsed_yaml['tags'] = ['fact_note']
    elif not isinstance(parsed_yaml['tags'], list):
        if isinstance(parsed_yaml['tags'], str):
            parsed_yaml['tags'] = [t.strip() for t in re.split(r'[,\s]+', parsed_yaml['tags']) if t.strip()]
        else:
            parsed_yaml['tags'] = ['fact_note']
            
    # Ensure 'fact_note' is in tags
    if 'fact_note' not in parsed_yaml['tags']:
        parsed_yaml['tags'].insert(0, 'fact_note')
        
    # Format tags to Obsidian requirements (separate lines starting with - )
    new_yaml_lines = ['tags:']
    for tag in parsed_yaml['tags']:
        new_yaml_lines.append(f"  - {tag}")
        
    # Write other key-value pairs
    for k, v in parsed_yaml.items():
        if k == 'tags':
            continue
        try:
            val_dump = yaml.safe_dump({k: v}, default_flow_style=False, allow_unicode=True).strip()
            new_yaml_lines.append(val_dump)
        except Exception:
            new_yaml_lines.append(f"{k}: {v}")
            
    final_content = "\n".join(['---'] + new_yaml_lines + ['---'] + lines[closing_idx+1:])
    return final_content

def save_state_incrementally(state_path, newly_processed_item):
    try:
        state_dir = os.path.dirname(state_path)
        if state_dir and not os.path.exists(state_dir):
            os.makedirs(state_dir)
            
        # Re-read state from disk to merge with concurrent changes
        current_state = []
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    current_state = json.load(f)
                    if not isinstance(current_state, list):
                        current_state = []
            except Exception as e:
                logger.warning(f"Failed to read state during merge: {e}")
                
        if newly_processed_item not in current_state:
            current_state.append(newly_processed_item)
            
        tmp_path = state_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(current_state, f, indent=2, ensure_ascii=False)
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, state_path)
                break
            except PermissionError as pe:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to replace state path after {max_retries} attempts: {pe}")
                    raise pe
                backoff = 0.1 * (2 ** attempt)
                logger.warning(f"PermissionError on replacing state path. Retrying in {backoff:.2f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(backoff)
        logger.debug(f"Saved state incrementally for {newly_processed_item} to {state_path}")
        return current_state
    except Exception as e:
        logger.error(f"Error saving state to {state_path}: {e}")
        return None

def main(args=None):
    parsed_args = parse_args(args)
    
    # Configure logging
    configure_logging(parsed_args.log_path, parsed_args.verbose)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)
    
    # Resolve default paths
    pdf_dir = parsed_args.pdf_dir
    if not pdf_dir:
        pdf_dir = os.path.join(workspace_root, "raw", "sources")
    else:
        pdf_dir = os.path.abspath(pdf_dir)
        
    output_dir = parsed_args.output_dir
    if not output_dir:
        output_dir = os.path.join(workspace_root, "wiki")
    else:
        output_dir = os.path.abspath(output_dir)
        
    state_path = parsed_args.state_path
    if not state_path:
        state_path = os.path.join(workspace_root, "raw", "sources", "processed_papers.json")
    else:
        state_path = os.path.abspath(state_path)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load processed papers state
    processed = []
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                processed = json.load(f)
                if not isinstance(processed, list):
                    processed = []
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            
    # Find PDFs
    pdf_files = []
    if os.path.isfile(pdf_dir):
        pdf_files = [pdf_dir]
    elif os.path.isdir(pdf_dir):
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
    logger.info(f"Found {len(pdf_files)} PDF files.")
    
    # Load Prompt Template (dynamic path based on CLI arg)
    prompt_path = parsed_args.prompt_path
    if not os.path.isabs(prompt_path):
        candidate_workspace = os.path.join(workspace_root, prompt_path)
        if os.path.exists(candidate_workspace):
            prompt_path = candidate_workspace
        else:
            prompt_path = os.path.abspath(prompt_path)
    else:
        prompt_path = os.path.abspath(prompt_path)
        
    logger.info(f"Loading prompt template from: {prompt_path}")
    system_instruction_text = (
        "Role: Strict Academic Empiricist\n"
        "Mission:\n"
        "1. Read the provided PDF document.\n"
        "2. Your ONLY goal is objective extraction. Extract the original author's explicit hypotheses, methodologies, empirical findings, and stated limitations.\n"
        "3. DO NOT inject external theories, metaphors, or your own interpretations. Remain 100% faithful to the source text.\n"
        "4. Write a markdown note.\n"
        "5. You MUST include YAML frontmatter at the top with tags.\n"
    )
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_instruction_text = f.read()
    except Exception as e:
        logger.warning(f"Could not load prompt template from {prompt_path}: {e}. Using default system instruction.")
        
    # Configure Gemini Model with system_instruction
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "dummy-key"))
    model = genai.GenerativeModel(
        model_name=parsed_args.model_name,
        system_instruction=system_instruction_text
    )
    
    new_processed = list(processed)
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        
        # Concurrent execution safety: re-check latest state from disk before processing
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    latest_state = json.load(f)
                    if isinstance(latest_state, list) and filename in latest_state and not parsed_args.force:
                        logger.info(f"Skipping already processed paper (concurrent check): {filename}")
                        continue
            except Exception:
                pass
                
        if filename in new_processed and not parsed_args.force:
            logger.info(f"Skipping already processed paper: {filename}")
            continue
            
        logger.info(f"Processing: {filename}")
        try:
            # Extract text with PDF read guards
            pdf_text = extract_text_from_pdf(pdf_path)
        except Exception as e:
            logger.error(f"Error reading PDF {filename}: {e}. Skipping file.")
            continue
            
        # Check length of extracted text (WITHOUT any pytest bypass)
        if len(pdf_text.strip()) < 100:
            logger.warning(f"Extracted text from {filename} is too short ({len(pdf_text)} characters). Skipping (likely scanned or empty).")
            continue
            
        try:
            # Invoke LLM using the model configured with system_instruction
            prompt = f"Extract facts for: {filename}\nContent:\n{pdf_text}"
            
            response = model.generate_content(prompt)
            note_content = response.text
            
            # Clean and format response
            cleaned_content = clean_and_format_markdown(note_content)
            
            # Save Fact note file
            title = os.path.splitext(filename)[0]
            fact_note_filename = f"Fact - {title}.md"
            fact_note_path = os.path.join(output_dir, fact_note_filename)
            
            with open(fact_note_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
                
            # Save state incrementally (merges automatically)
            updated_state = save_state_incrementally(state_path, filename)
            if updated_state is not None:
                new_processed = updated_state
            else:
                if filename not in new_processed:
                    new_processed.append(filename)
            
            logger.info(f"Generated Fact note: {fact_note_filename}")
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            
    logger.info("Fact extraction completed.")
    return 0

def extract_text_from_pdf(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
            if len(reader.pages) > 0:
                _ = reader.pages[0]
            else:
                raise ValueError("PDF has no pages after decryption")
        except Exception as e:
            raise ValueError(f"PDF is encrypted and decryption with empty password failed: {e}")
            
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

if __name__ == "__main__":
    sys.exit(main())
