# ⚖️ The 3-Tier Epistemological Harness

This document defines the absolute rules (The Harness) for how information MUST be extracted and connected within this Zettelkasten system. Do not deviate from these rules, regardless of whether you are running in Mode A (Native Swarm), Mode B (Self-Debate), or Mode C (Python Script).

## 🛑 Rule 1: The Separation of Extraction and Synthesis
To prevent AI hallucination and distortion of the original authors' intent, the system strictly separates **Fact Extraction (Layer 1)** from **Knowledge Synthesis (Layer 2)**.
- **Layer 1 (Extraction):** When processing a new raw PDF, the AI must ONLY act as a strict Empiricist. It must extract the original hypotheses, methodologies, data, and conclusions exactly as the authors intended. **No external lenses or metaphors are allowed here.**
- **Layer 2 (Synthesis):** When connecting papers, the AI must NOT rely on the authors' explicit connections (which are often missing). Instead, the AI must use rigorous, multi-agent debate to discover latent, implicit correlations between the extracted facts of different papers.

## 🛑 Rule 2: Correlation-Based Concept Generation
Concept Hubs (`wiki/concepts/`) cannot be generated arbitrarily. 
- A Concept Hub can ONLY be created if it serves as a mathematical, statistical, or deep theoretical intersection (a Correlation Hub) between TWO OR MORE Fact Notes.
- E.g., If Paper A talks about "Autonomy" and Paper B talks about "Turnover", the Concept Hub must be the specific causal correlation between the two: "The Interaction Effect of Autonomy on Turnover Rates."

## 🛑 Rule 3: The Topological Mandate
- Every Fact Note (`Fact - [Title].md`) must be hyper-linked (`[[ ]]`) to at least one Concept Hub.
- Every Concept Hub must explicitly cite the specific Fact Notes that form its foundation.
- This bidirectional linking guarantees a fully traceable, rigorous academic knowledge graph.

## 🛑 Rule 4: Dynamic Persona Generation (동적 페르소나 소환)
Whether running a native swarm, an automated Python backend, or using the manual prompt templates, the AI must dynamically adopt multiple expert personas. 
- Do not rely on a single perspective. Automatically generate and embody specialized personas (e.g., "Quantum Physicist", "Behavioral Economist", "Devil's Advocate") relevant to the papers' domain.
- Force these personas to debate internally (Self-Debate) or externally (Multi-Agent) to cross-validate findings and stress-test assumptions.

## 🛑 Rule 5: Cross-Validation & Micro-Incommensurability
- Never just summarize or list facts side-by-side.
- The system must ruthlessly seek out **anomalies, contradictions, and micro-incommensurabilities** between different papers. Use brute-force cross-validation to discover hidden causations that neither author explicitly stated.
