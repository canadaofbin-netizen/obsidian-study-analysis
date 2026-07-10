# 🚀 Infinite Research Swarm: AI Autonomous Zettelkasten

Welcome to the **Infinite Research Swarm**, a powerful Dual-Mode AI framework that turns a folder of raw PDFs into a hyper-linked, deeply analyzed knowledge graph. 

This is not just a template. It is an architecture designed to ruthlessly seek out anomalies, force multi-agent debates, and discover latent correlations across hundreds of papers.

## ⚖️ The Dual-Mode Execution

You can run this framework in one of two ways, depending on your technical comfort:

### ⚙️ Mode A: Multi-Agent Swarm (Automated)
For developers with API keys (e.g., Gemini). You can run the Python backend in the `scripts/` folder to spawn an actual swarm of AI agents.
- Automatically download papers from arXiv (`download_papers.py`).
- Agents autonomously extract facts and run infinite debates (`run_debate.py`).
- Perfect for analyzing 100+ papers while you sleep.

### 🧠 Mode B: Manual Self-Debate (Prompt-based)
For users without automated API setups. You can simply copy-paste the highly optimized prompts in the `prompts/` folder into standard LLMs (ChatGPT, Claude, etc.).
- The prompts are injected with a **"Self-Debate"** engine. The LLM will dynamically split its own consciousness into multiple expert personas (e.g., "Meta-Analyst", "Devil's Advocate") and argue with itself to give you the exact same rigorous results as a multi-agent swarm.
- Prompts include a user choice: you can ask the AI to format the output strictly for Obsidian, or as standard readable markdown.

## 🛑 The Core Philosophy (See `SYSTEM_HARNESS.md`)
1. **Separation of Fact & Synthesis**: Facts are extracted with 100% empiricism. Synthesis is generated through brutal debate.
2. **Micro-Incommensurability**: The system doesn't just summarize; it actively hunts for contradictions and anomalies between papers.
3. **Topological Mandate**: All findings must be mathematically linked `[[wikilinks]]` in Obsidian to create a traceable graph.

## 📁 Repository Structure
- `raw/sources/`: Drop your raw PDFs here.
- `wiki/`: Where purely objective `Fact - *.md` notes live.
- `wiki/concepts/`: Where overarching theoretical Correlation Hubs are generated.
- `wiki/synthesis/`: Debate transcripts and executive summaries.
- `scripts/`: Python backend for Mode A (Automated Swarm).
- `prompts/`: Advanced Self-Debate Prompts for Mode B (Manual).

## 🛠️ Getting Started
1. Clone this repository and open it as an Obsidian Vault.
2. Install the **Dataview** plugin in Obsidian to activate `Dashboard.md`.
3. Choose your path:
   - **Mode A:** Set up your API keys and run the Python scripts in `scripts/`.
   - **Mode B:** Drop PDFs in `raw/sources/`, and use the `prompts/1_fact_extractor.md` and `prompts/2_academic_orchestrator.md` inside your favorite LLM.
