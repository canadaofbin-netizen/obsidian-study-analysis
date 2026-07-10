# 🚀 Infinite Research Swarm: AI Autonomous Zettelkasten

Welcome to the **Infinite Research Swarm**, a powerful AI framework that turns a folder of raw PDFs into a hyper-linked, deeply analyzed knowledge graph. 

This is not just a template. It is an architecture designed to ruthlessly seek out anomalies, force multi-agent debates, and discover latent correlations across hundreds of papers.

## ⚖️ The 3-Tier Execution Architecture

Depending on your environment and technical expertise, you can run this framework in one of three ways:

### 🌟 Mode A: Native Multi-Agent Swarm (Maximum Efficiency)
For users operating within an Advanced Agentic AI Workspace (like Antigravity). 
- The AI natively summons multiple subagents that run entirely in the background.
- They autonomously read PDFs, extract facts, debate, and write directly to your Obsidian vault.
- **Zero Python required, Zero external API keys needed.** Maximum efficiency and full automation.

### 🧠 Mode B: Manual Self-Debate (Prompt-based)
For users without automated agent infrastructures. Simply copy-paste the highly optimized prompts in the `prompts/` folder into standard LLMs (ChatGPT, Claude, etc.).
- The prompts are injected with a **"Self-Debate"** engine. The LLM will dynamically split its own consciousness into multiple expert personas (e.g., "Meta-Analyst", "Devil's Advocate") and argue with itself.
- Provides rigorous results identical to a multi-agent swarm, just within a single chat window.

### ⚙️ Mode C: Legacy Python Script (Automated but requires API)
For traditional developers who want to run cron jobs on their own servers. 
- You can run the Python backend in the `scripts/` folder (`run_debate.py`).
- **Requires an external Google API Key (e.g. Gemini).** 
- Automatically downloads papers from arXiv and orchestrates debates via API calls.

## 🛑 The Core Philosophy (See `SYSTEM_HARNESS.md`)
1. **Separation of Fact & Synthesis**: Facts are extracted with 100% empiricism. Synthesis is generated through brutal debate.
2. **Micro-Incommensurability**: The system doesn't just summarize; it actively hunts for contradictions and anomalies between papers.
3. **Topological Mandate**: All findings must be mathematically linked `[[wikilinks]]` in Obsidian to create a traceable graph.

## 📁 Repository Structure
- `raw/sources/`: Drop your raw PDFs here.
- `wiki/`: Where purely objective `Fact - *.md` notes live.
- `wiki/concepts/`: Where overarching theoretical Correlation Hubs are generated.
- `wiki/synthesis/`: Debate transcripts and executive summaries.
- `scripts/`: Python backend for Mode C.
- `prompts/`: Advanced Self-Debate Prompts for Mode B.

## 🛠️ Getting Started
1. Clone this repository and open it as an Obsidian Vault.
2. Install the **Dataview** plugin in Obsidian to activate `Dashboard.md`.
3. Choose your path (Mode A, Mode B, or Mode C) and begin your research!
