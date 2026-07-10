import argparse
import os
import sys
import logging

# Ensure scripts directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_papers import download_arxiv_papers
from extract_facts import main as run_extraction
from run_debate import main as run_debate
from compliance_checker import check_vault

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Integration")
    parser.add_argument('--query', type=str, default='all:"AI Agents"')
    parser.add_argument('--max-results', type=int, default=100)
    parser.add_argument('--pdf-dir', type=str, default=None)
    parser.add_argument('--wiki-dir', type=str, default=None)
    parser.add_argument('--synthesis-dir', type=str, default=None)
    parser.add_argument('--concepts-dir', type=str, default=None)
    parser.add_argument('--state-path', type=str, default=None)
    parser.add_argument('--registry-path', type=str, default=None)
    parser.add_argument('--log-path', type=str, default=None)
    return parser.parse_args(args)

def main(args=None):
    parsed_args = parse_args(args)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(script_dir)
    
    # Resolve default paths
    pdf_dir = parsed_args.pdf_dir or os.path.join(workspace_root, "raw", "sources")
    wiki_dir = parsed_args.wiki_dir or os.path.join(workspace_root, "wiki")
    synthesis_dir = parsed_args.synthesis_dir or os.path.join(wiki_dir, "synthesis")
    concepts_dir = parsed_args.concepts_dir or os.path.join(wiki_dir, "concepts")
    state_path = parsed_args.state_path or os.path.join(pdf_dir, "processed_papers.json")
    registry_path = parsed_args.registry_path or os.path.join(pdf_dir, "download_registry.json")
    log_path = parsed_args.log_path or os.path.join(workspace_root, "pipeline.log")
    
    # Setup logging
    logger = logging.getLogger("Pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = True
    for h in list(logger.handlers):
        logger.removeHandler(h)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logger.info("Pipeline started.")
    
    # Ensure directories exist
    for d in [pdf_dir, wiki_dir, synthesis_dir, concepts_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            
    # Step 1: Ingest
    logger.info("Step 1: Running Paper Ingestion...")
    try:
        download_arxiv_papers(
            query=parsed_args.query,
            max_results=parsed_args.max_results,
            output_dir=pdf_dir,
            registry_path=registry_path
        )
        logger.info("Ingestion completed successfully.")
    except SystemExit as e:
        if e.code != 0:
            logger.error(f"Ingestion exited with non-zero code: {e.code}")
            sys.exit(e.code)
        logger.info("Ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)
        
    # Step 2: Extract
    logger.info("Step 2: Running Fact Extraction...")
    try:
        ret = run_extraction([
            '--pdf-dir', pdf_dir,
            '--output-dir', wiki_dir,
            '--state-path', state_path
        ])
        if ret is not None and ret != 0:
            raise RuntimeError(f"Fact extraction failed with status code: {ret}")
        logger.info("Fact Extraction completed successfully.")
    except BaseException as e:
        logger.error(f"Fact Extraction failed: {e}")
        sys.exit(1)
        
    # Step 3: Debate
    logger.info("Step 3: Running Colosseum Debate...")
    try:
        run_debate([
            '--wiki-dir', wiki_dir,
            '--synthesis-dir', synthesis_dir,
            '--concepts-dir', concepts_dir
        ])
        logger.info("Colosseum Debate completed successfully.")
    except BaseException as e:
        logger.error(f"Colosseum Debate failed: {e}")
        sys.exit(1)
        
    # Step 4: Compliance Validation
    logger.info("Step 4: Running Compliance Validation...")
    violations = check_vault(wiki_dir)
    if violations:
        logger.error("Compliance violations found:")
        for file, file_violations in violations.items():
            logger.error(f"  File: {file}")
            for v in file_violations:
                logger.error(f"    - {v}")
        logger.error("Pipeline failed compliance check.")
        sys.exit(1)
    else:
        logger.info("Vault compliance validation passed successfully.")
        
    logger.info("Pipeline executed and finished successfully.")

if __name__ == "__main__":
    main()
