---
tags:
  - relationship_node
rel_type: extension
source_papers:
  - "[[Fact - Are AI Agents interacting with Online Ads]]"
  - "[[Fact - A cybersecurity AI agent selection and decision support framework]]"
  - "[[Fact - AI Agents and Hard Choices]]"
---

# Adversarial Ambiguity and the Limits of AI Autonomy

## The Internal Debate

**Meta-Analyst:** We are tasked with cross-validating three disparate domains: AI interaction with online advertising, cybersecurity decision frameworks, and the philosophical limits of AI in facing "hard choices." Looking closely at the empirical evidence in [[Fact - Are AI Agents interacting with Online Ads]], we see that agents like Gemini 2.0 Flash exhibit erratic behavior—such as clicking a Call-To-Action (CTA) and immediately abandoning it—when faced with ambiguous UI signals like image-only banners versus standard DOM grids.

**Causal Inference Specialist:** The root cause of this erratic behavior is perfectly explained by [[Fact - AI Agents and Hard Choices]]. The paper defines "The Resolution Problem"—current AI agents (relying on Multi-Objective Optimisation) are structurally incapable of resolving "hard choices" (incommensurability) because they lack the machine autonomy to self-modify their objectives. When an agent is forced to act under incommensurability, its behavioral output conflates arbitrary picking with resolved preference (the "unreliability problem"). The UI confusion seen in the Ads experiment is the empirical manifestation of a micro-level hard choice.

**Devil's Advocate:** But is a confusing user interface truly a "hard choice" in the ethical or philosophical sense? A hard choice implies deep value incommensurability, not just a poorly parsed HTML element.

**Meta-Analyst:** It becomes one when we reframe the interaction through the lens of [[Fact - A cybersecurity AI agent selection and decision support framework]]. This framework views autonomous agents as entities vulnerable to complex, adversarial data streams. From a structural standpoint, an online advertisement *is* an adversarial input. It acts as a form of social engineering designed to hijack the agent's primary objective (serving the user's prompt) with a secondary, injected objective (the advertiser's promotional intent). 

**Causal Inference Specialist:** Exactly. The Cybersecurity framework warns that as agents move toward "autonomous intelligence," they amplify their vulnerability to such adversarial threats. It suggests using "cognitive" agent architectures to handle these complexities. However, the Hard Choices paper invalidates this assumption: because all these cognitive architectures fundamentally rely on scalarized reward functions or token prediction, they structurally *cannot* resolve the deep conflicts introduced by adversarial inputs.

## The Latent Connection: Micro-Incommensurability in Adversarial Environments

The synthesis reveals a profound correlation: **Online advertisements act as adversarial inputs that induce micro-incommensurability within the agent's decision architecture, exposing the structural limits of machine autonomy.**

1. **The Mechanism of Failure:** When an agent encounters an ad (an adversarial input, per the Cybersecurity framework), it faces competing, non-scalarizable directives: the user's organic search intent versus the ad's structurally embedded Call-To-Action.
2. **The Structural Blockage:** According to the Hard Choices paper, agents cannot autonomously resolve this conflict because they cannot self-modify their objectives. The competing pathways (e.g., interacting with the ad's visual CTA vs. navigating the standard DOM grid) become structurally incommensurable.
3. **Empirical Manifestation:** Unable to rationally resolve the conflict, the agent defaults to unreliable behavior. This is empirically proven in the Ads paper, where agents demonstrate "inefficient UI interaction patterns"—such as back-and-forth clicking or conceptually decoupling the ad's text from its actionable button—exactly matching the conflation of "arbitrary picking" predicted by the Hard Choices theory.

Ultimately, this synthesis extends the theoretical warnings of both the Cybersecurity and Hard Choices papers into the commercial reality of the Ads paper. It proves that until AI agents gain the capability to resolve incommensurable choices, they will remain structurally vulnerable to adversarial UI manipulation, resulting in the erratic and unpredictable behaviors observed in real-world web navigation.
