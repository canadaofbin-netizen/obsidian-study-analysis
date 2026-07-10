import os
import sys
import time
import json
import argparse
import re
import logging
import socket
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("download_papers")

def extract_arxiv_id(url):
    """
    Extract identifier from a URL like:
    http://arxiv.org/abs/2109.00001v1 -> 2109.00001
    http://arxiv.org/abs/math/0211111v1 -> math/0211111
    """
    if not url:
        return None
    url = url.split('?')[0].split('#')[0]
    # Look for /abs/ or /pdf/ separators
    for separator in ['/abs/', '/pdf/']:
        if separator in url:
            id_part = url.split(separator)[-1]
            # Strip version suffix like v1, v2
            id_part = re.sub(r'v\d+$', '', id_part)
            return id_part
            
    # Fallback: take the last part of the URL
    parts = [p for p in url.split('/') if p]
    if not parts:
        return None
    last_segment = parts[-1]
    last_segment = re.sub(r'v\d+$', '', last_segment)
    
    # Check if the segment before it is an archive name (e.g., math, hep-th)
    if len(parts) >= 2 and re.match(r'^[a-z\-]+$', parts[-2]):
        return f"{parts[-2]}/{last_segment}"
        
    return last_segment

def load_registry(registry_path):
    """Load registry file, starting fresh if not found or invalid."""
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                logger.warning("Registry format is invalid (not a dict). Starting fresh.")
        except Exception as e:
            logger.warning(f"Failed to load registry from {registry_path}: {e}. Starting fresh.")
    return {}

def save_registry(registry_path, registry):
    """Save registry file atomically."""
    try:
        # Ensure registry directory exists
        os.makedirs(os.path.dirname(os.path.abspath(registry_path)), exist_ok=True)
        tmp_registry = registry_path + ".tmp"
        with open(tmp_registry, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        os.replace(tmp_registry, registry_path)
    except Exception as e:
        logger.error(f"Failed to save registry to {registry_path}: {e}")

def make_request_with_retry(url, is_download=False, filepath=None, timeout=30):
    """
    Perform HTTP request with a retry loop (up to 3 attempts) and exponential backoff.
    Specifically handles HTTP 503, socket timeouts, and other transient network issues.
    """
    max_attempts = 3
    # Set default socket timeout for urllib
    socket.setdefaulttimeout(timeout)
    
    for attempt in range(max_attempts):
        try:
            if is_download:
                # Ensure output directory exists
                tmp_filepath = filepath + ".tmp"
                os.makedirs(os.path.dirname(os.path.abspath(tmp_filepath)), exist_ok=True)
                
                logger.info(f"Downloading PDF (attempt {attempt + 1}/{max_attempts}): {url}")
                try:
                    urllib.request.urlretrieve(url, tmp_filepath)
                    # Atomically rename
                    os.replace(tmp_filepath, filepath)
                finally:
                    if os.path.exists(tmp_filepath):
                        try:
                            os.remove(tmp_filepath)
                        except Exception as cleanup_err:
                            logger.warning(f"Failed to clean up temp file {tmp_filepath}: {cleanup_err}")
                # Enforce delay between requests (3 seconds)
                time.sleep(3)
                return True
            else:
                logger.info(f"Querying arXiv API (attempt {attempt + 1}/{max_attempts}): {url}")
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    data = response.read()
                # Enforce delay between requests (3 seconds)
                time.sleep(3)
                return data
        except Exception as e:
            # Let local filesystem/disk errors propagate immediately without retrying
            if isinstance(e, (PermissionError, FileNotFoundError, IsADirectoryError, NotADirectoryError, FileExistsError)):
                raise e
            if not isinstance(e, (urllib.error.URLError, TimeoutError, ConnectionError)):
                raise e

            is_503 = False
            if isinstance(e, urllib.error.HTTPError):
                if e.code == 503:
                    is_503 = True
            
            logger.warning(f"Request failed (attempt {attempt + 1}/{max_attempts}): {e}" + (" [HTTP 503 Service Unavailable]" if is_503 else ""))
            
            if attempt == max_attempts - 1:
                # Re-raise on last attempt failure
                raise e
            
            # Exponential backoff: 3s, 6s, 12s
            backoff = 3 * (2 ** attempt)
            logger.info(f"Waiting {backoff}s before retrying...")
            time.sleep(backoff)

def download_arxiv_papers(query, max_results, output_dir, registry_path):
    """Query arXiv API and download PDFs with registry-based duplicate check."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    registry = load_registry(registry_path)
    
    # Build query URL
    quoted_query = urllib.parse.quote(query)
    url = f'http://export.arxiv.org/api/query?search_query={quoted_query}&start=0&max_results={max_results}'
    
    # Query metadata
    try:
        data = make_request_with_retry(url, is_download=False)
        root = ET.fromstring(data)
    except Exception as e:
        logger.error(f"Critical error fetching metadata from arXiv: {e}")
        raise RuntimeError(f"Critical error fetching metadata from arXiv: {e}") from e
        
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('atom:entry', ns)
    logger.info(f"Found {len(entries)} papers. Starting download...")
    
    download_failures = 0
    
    for i, entry in enumerate(entries, 1):
        title_node = entry.find('atom:title', ns)
        if title_node is None or not title_node.text:
            logger.warning(f"[{i}/{len(entries)}] Skip: Missing title.")
            continue
            
        title = title_node.text.replace('\n', ' ').strip()
        clean_title = "".join(c for c in title if c.isalnum() or c in " _-")[:100]
        
        # Get PDF url
        pdf_url = None
        for link in entry.findall('atom:link', ns):
            if link.attrib.get('title') == 'pdf' or link.attrib.get('type') == 'application/pdf':
                pdf_url = link.attrib.get('href')
                break
                
        if not pdf_url:
            logger.warning(f"[{i}/{len(entries)}] No PDF link found for: {title}")
            continue
            
        # Get arXiv ID from entry ID or pdf url
        id_tag = entry.find('atom:id', ns)
        id_url = id_tag.text.strip() if id_tag is not None else None
        if not id_url:
            id_url = pdf_url
            
        arxiv_id = extract_arxiv_id(id_url)
        
        pdf_filename = f"{clean_title}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)
        
        # Duplicate Checks
        in_registry = arxiv_id in registry if arxiv_id else False
        file_exists = os.path.exists(pdf_path)
        
        if in_registry:
            logger.info(f"[{i}/{len(entries)}] Skip (Already registered): {clean_title} ({arxiv_id})")
            continue
            
        if file_exists:
            logger.info(f"[{i}/{len(entries)}] Skip (File exists on disk): {clean_title}")
            if arxiv_id:
                registry[arxiv_id] = {
                    "title": title,
                    "filename": pdf_filename,
                    "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                save_registry(registry_path, registry)
                logger.info(f"[{i}/{len(entries)}] Registered existing local paper: {arxiv_id}")
            continue
            
        # Download
        logger.info(f"[{i}/{len(entries)}] Downloading: {clean_title} ({arxiv_id})")
        if not pdf_url.endswith('.pdf'):
            pdf_url += '.pdf'
            
        try:
            make_request_with_retry(pdf_url, is_download=True, filepath=pdf_path)
            
            # Record in registry upon successful download
            if arxiv_id:
                registry[arxiv_id] = {
                    "title": title,
                    "filename": pdf_filename,
                    "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                save_registry(registry_path, registry)
        except Exception as e:
            logger.error(f"Failed to download {clean_title} ({arxiv_id}) after retries: {e}")
            download_failures += 1
            
    logger.info("Download process completed!")
    if download_failures > 0:
        logger.error(f"{download_failures} downloads failed.")
        raise RuntimeError(f"{download_failures} downloads failed.")
    else:
        return registry

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    default_output_dir = os.path.join(workspace_root, "raw", "sources")
    default_registry_path = os.path.join(default_output_dir, "download_registry.json")
    
    parser = argparse.ArgumentParser(description="Download research papers from arXiv API.")
    parser.add_argument("--query", type=str, default='all:"AI Agents"', help="Query for arXiv API (default: 'all:\"AI Agents\"')")
    parser.add_argument("--max-results", type=int, default=100, help="Max papers to download (default: 100)")
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="Target directory path (default: raw/sources)")
    parser.add_argument("--registry-path", type=str, default=default_registry_path, help="Path to download_registry.json")
    
    args = parser.parse_args()
    
    # Resolve relative paths relative to current working directory (or workspace root if requested)
    # The absolute path functions handles this.
    output_dir = os.path.abspath(args.output_dir)
    registry_path = os.path.abspath(args.registry_path)
    
    try:
        download_arxiv_papers(
            query=args.query,
            max_results=args.max_results,
            output_dir=output_dir,
            registry_path=registry_path
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical execution error: {e}")
        sys.exit(1)
