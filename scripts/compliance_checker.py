import argparse
import os
import re
import sys
import yaml

def check_file(file_path):
    violations = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    content = content.lstrip('\ufeff')
    lines = content.split('\n')
    
    # 1. Check YAML frontmatter boundaries
    first_line = lines[0].rstrip('\r') if lines else ""
    if not lines or first_line != '---':
        violations.append("File does not start with YAML boundary '---'")
        
    closing_idx = -1
    for idx, line in enumerate(lines[1:], start=1):
        if line.rstrip('\r') == '---':
            closing_idx = idx
            break
            
    if closing_idx == -1:
        violations.append("File does not close the YAML boundary with '---'")
    else:
        # Extract YAML frontmatter block
        yaml_lines = [line.rstrip('\r') for line in lines[1:closing_idx]]
        yaml_block = "\n".join(yaml_lines)
        try:
            parsed_yaml = yaml.safe_load(yaml_block)
            if not isinstance(parsed_yaml, dict):
                violations.append("YAML frontmatter is not a dictionary/mapping")
            else:
                # Check tags is a list
                if 'tags' in parsed_yaml:
                    tags = parsed_yaml['tags']
                    if not isinstance(tags, list):
                        violations.append("YAML 'tags' field is not a list/sequence")
                    else:
                        # Verify tag items are non-empty strings
                        for tag in tags:
                            if not isinstance(tag, str) or not tag.strip():
                                violations.append("YAML 'tags' list contains invalid (non-string or empty) items")
                                break
                        
                        # Additionally, check that if block format is used, tags start with a dash on a new line.
                        for i, line in enumerate(lines[1:closing_idx], start=1):
                            stripped = line.rstrip('\r').strip()
                            if stripped.startswith('tags:'):
                                suffix = stripped.split('tags:', 1)[1].strip()
                                if suffix.startswith('#'):
                                    suffix = ""
                                if not suffix:
                                    # Block format is used. Verify subsequent lines start with '-'
                                    has_items = False
                                    for next_line in lines[i + 1:closing_idx]:
                                        stripped_next = next_line.rstrip('\r').strip()
                                        if not stripped_next or stripped_next.startswith('#'):
                                            continue
                                        if stripped_next.startswith('-'):
                                            has_items = True
                                            break
                                        else:
                                            break
                                    if not has_items and len(tags) > 0:
                                        violations.append("YAML 'tags' list items do not start with '-' on new lines")
                                break
        except Exception as e:
            violations.append(f"Failed to parse YAML frontmatter: {e}")
            
    # Create clean_content by replacing the frontmatter lines with spaces
    # of the same length, keeping the newlines.
    if lines and first_line == '---' and closing_idx != -1:
        body_lines = list(lines)
        for i in range(closing_idx + 1):
            line = body_lines[i]
            if line.endswith('\r'):
                body_lines[i] = ' ' * (len(line) - 1) + '\r'
            else:
                body_lines[i] = ' ' * len(line)
        clean_content = '\n'.join(body_lines)
    else:
        clean_content = '\n'.join(lines)
        
    # Strip HTML and Obsidian comments, fenced code blocks and inline code
    def strip_keeping_newlines(match):
        text = match.group(0)
        return re.sub(r'[^\r\n]', ' ', text)
        
    clean_content = re.sub(r'<!--.*?-->', strip_keeping_newlines, clean_content, flags=re.DOTALL)
    clean_content = re.sub(r'%%.*?%%', strip_keeping_newlines, clean_content, flags=re.DOTALL)
    
    # Strip fenced code blocks using a robust line-by-line state machine
    lines_to_process = clean_content.split('\n')
    in_code_block = False
    code_block_char = None
    code_block_len = 0
    for idx, line in enumerate(lines_to_process):
        stripped = line.rstrip('\r').strip()
        if not in_code_block:
            backtick_match = re.match(r'^(\s*)```+', line)
            tilde_match = re.match(r'^(\s*)~~~+', line)
            if backtick_match:
                in_code_block = True
                code_block_char = '`'
                code_block_len = len(backtick_match.group(0).strip())
                lines_to_process[idx] = re.sub(r'[^\r]', ' ', line)
            elif tilde_match:
                in_code_block = True
                code_block_char = '~'
                code_block_len = len(tilde_match.group(0).strip())
                lines_to_process[idx] = re.sub(r'[^\r]', ' ', line)
        else:
            pattern = r'^\s*' + (re.escape(code_block_char) * code_block_len) + r'+\s*$'
            if re.match(pattern, stripped):
                in_code_block = False
                lines_to_process[idx] = re.sub(r'[^\r]', ' ', line)
            else:
                lines_to_process[idx] = re.sub(r'[^\r]', ' ', line)
    if in_code_block:
        violations.append("File contains an unclosed code block fence")
    clean_content = '\n'.join(lines_to_process)
    
    clean_content = re.sub(r'`+[^`\r\n]+`+', strip_keeping_newlines, clean_content)
    
    # 2. Check WikiLinks syntax via character stack parsing
    open_stack = []
    i = 0
    n = len(clean_content)
    while i < n:
        if i + 1 < n and clean_content[i:i+2] == '[[':
            if open_stack:
                violations.append("Mismatched wikilink brackets: Nested wikilinks are not allowed (e.g., [[foo[[bar]]])")
            open_stack.append(i)
            i += 2
        elif i + 1 < n and clean_content[i:i+2] == ']]':
            if not open_stack:
                violations.append("Mismatched wikilink brackets: orphaned close brackets ']]' found")
            else:
                start_idx = open_stack.pop()
                link_content = clean_content[start_idx+2:i]
                if not link_content.strip():
                    violations.append("Empty wikilink '[[]]' found")
                elif '\n' in link_content:
                    violations.append("Mismatched wikilink brackets: cross-line wikilinks are not allowed")
                elif '[' in link_content or ']' in link_content:
                    violations.append("Mismatched wikilink brackets: malformed wikilink brackets found")
            i += 2
        else:
            i += 1
            
    if open_stack:
        violations.append("Mismatched wikilink brackets: orphaned open brackets '[[' found")
        
    # 3. Document rendering integrity (headers missing space after '#')
    clean_lines = clean_content.split('\n')
    for idx, line in enumerate(clean_lines, start=1):
        # Normalize leading spaces and blockquotes
        stripped = line
        prev = None
        while stripped != prev:
            prev = stripped
            stripped = stripped.strip()
            if stripped.startswith('>'):
                stripped = stripped[1:]
                continue
            m = re.match(r'^([-*+]|\d+\.)\s+(.*)', stripped)
            if m:
                stripped = m.group(2)
                continue
                
        stripped = stripped.strip()
        if stripped.startswith('#'):
            num_hashes = len(stripped) - len(stripped.lstrip('#'))
            if 1 <= num_hashes <= 6:
                if len(stripped) > num_hashes and stripped[num_hashes] != ' ':
                    violations.append(f"Line {idx}: Heading format is incorrect (missing space after '#')")
                    
    return violations

def check_vault(vault_dir):
    all_violations = {}
    
    ignored_dirs = {
        '.obsidian', '.git', '.pytest_cache', '__pycache__', 'raw',
        '.agents', 'prompts', '.venv', 'venv', 'node_modules', 'tests'
    }
    
    for root, dirs, files in os.walk(vault_dir):
        # Modify dirs in-place to prevent os.walk from entering ignored directories
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            if file.endswith('.md') and file not in {'Dashboard.md', 'README.md', 'SYSTEM_HARNESS.md', 'TEST_INFRA.md'}:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, vault_dir)
                violations = check_file(file_path)
                if violations:
                    all_violations[rel_path] = violations
                    
    return all_violations

def main(args=None):
    parser = argparse.ArgumentParser(description="Obsidian Compliance Checker")
    parser.add_argument('--vault-dir', type=str, default=None, help='Path to Obsidian vault directory')
    parsed_args = parser.parse_args(args)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)
    
    vault_dir = parsed_args.vault_dir or workspace_root
    
    print(f"Scanning vault for compliance: {vault_dir}")
    violations = check_vault(vault_dir)
    
    if violations:
        print("\nCompliance violations found:")
        for file, file_violations in violations.items():
            print(f"\nFile: {file}")
            for v in file_violations:
                print(f"  - {v}")
        print("\nCompliance Validation Failed.")
        sys.exit(1)
    else:
        print("Vault is fully compliant. No violations found.")
        sys.exit(0)

if __name__ == "__main__":
    main()
