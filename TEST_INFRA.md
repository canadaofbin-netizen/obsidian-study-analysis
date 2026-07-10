# E2E Test Infra: Infinite Research Swarm

## Test Philosophy
- Opaque-box, requirement-driven. Evaluates entry-point scripts without depending on implementation internals.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial + Real-World Workload Testing.
- Compliance: Validates output markdown notes against Obsidian vault syntax rules.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---------|---------------------|:------:|:------:|:------:|:------:|
| 1 | Ingestion (`download_papers.py`) | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 2 | Fact Extraction (`extract_facts.py`) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 3 | Colosseum Debate (`run_debate.py`) | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 4 | Obsidian Compliance Validator | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| 5 | E2E Integration Runner (`run_pipeline.py`) | PROJECT.md main runner | 5 | 5 | ✓ | ✓ |

## Test Architecture
- **Test Runner**: Pytest framework. Executed via `pytest tests/` from the workspace directory.
- **Mocks & Stubs**:
  - `urllib.request.urlopen` and `urllib.request.urlretrieve` are mocked to intercept calls to `export.arxiv.org` and return mock feeds / mock PDF file contents.
  - Gemini API (`google.generativeai`) is mocked using standard unittest mocking to return predefined mock fact extractions and synthetic debate transcripts, avoiding actual API costs and rate limit exceptions during tests.
- **Directory Layout**:
  - `obsidianfolder/tests/conftest.py` - Setup, global fixtures, and network mock patches.
  - `obsidianfolder/tests/test_infrastructure.py` - Verify mock frameworks and test setup.
  - `obsidianfolder/tests/test_tier1_feature_coverage.py` - Tier 1: 25 feature-level happy path tests (5 per feature).
  - `obsidianfolder/tests/test_tier2_boundary_corner.py` - Tier 2: 25 boundary/edge tests (5 per feature).
  - `obsidianfolder/tests/test_tier3_cross_feature.py` - Tier 3: 5 pairwise feature interaction tests.
  - `obsidianfolder/tests/test_tier4_real_world.py` - Tier 4: 5 full end-to-end workload execution scenarios.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Multi-Paper Pipeline Run | Ingest, Extract, Debate, Compliance | High |
| 2 | Vault Incremental Scan | Extraction, Debate, Duplicate Check | Medium |
| 3 | Network Error Recovery | Ingestion, Fault Tolerant Extraction | High |
| 4 | Large Vault Debate | Multi-persona debate, Concept Hub Update | High |
| 5 | Obsidian Compliance Sweep | Vault compliance checker execution | Medium |

## Coverage Thresholds
- Tier 1: 5 test cases per feature (Total 25)
- Tier 2: 5 boundary/corner test cases per feature (Total 25)
- Tier 3: Pairwise combination of all major pipeline steps (Total 5)
- Tier 4: 5 realistic application-level workloads (Total 5)
- Total E2E test cases: 60 cases
