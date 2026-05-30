# Discrete Diffused Stylistic Text

An advanced controllable text generation framework that creates persuasive, unifying, and contextually appropriate counterspeech against online hate speech. Instead of using generic or punitive measures (like bans or content removal), this project introduces **VQ-SEDD (Vector Quantized Score Entropy Discrete Diffusion)** to generate first-person counterspeech mirroring the distinct rhetorical styles and core philosophical values of historical moral leaders like **Mahatma Gandhi** and **Nelson Mandela**.

---

## 📌 Motivation & Problem Statement

### The Problem
Online hate speech is a massive, accelerating issue that compromises individual safety and corrupts public discourse. Conventional moderation tactics typically rely on **User Banning** or **Content Removal**. However, these techniques:
* Suffer from high latency and scaling boundaries.
* Frequently ignite fierce debates around the preservation of **Freedom of Speech**.
* Do not actively de-escalate underlying conflicts or rehabilitate perpetrators.

### The Solution: Personalized Counterspeech
Counterspeech offers a non-punitive, dialogue-driven alternative that undermines hate speech directly without silencing voices. Crucially, a **one-size-fits-all approach is highly ineffective**. To truly resonate, counterspeech requires a credible, authentic voice. 

This thesis answers a core question: **Can we automatically synthesize highly persuasive, dignified counterspeech modeled after history's greatest unifying figures?**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ HATE SPEECH INPUT:                                                          │
│ "Immigrants are ruining our country they should all be sent back!"           │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼ (VQ-SEDD Model)
┌──────────────────────────────────────────────────────────────────────────────┐
│ PERSONALIZED COUNTERSPEECH OUTPUT:                                           │
│ "My compatriot, I hear your words... Immigrants have played a vital role in  │
│ building our communities... Let us work together to ensure every citizen has │
│ the right to shape our collective destiny."                                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎭 Persona Profiling: Gandhi vs. Mandela

The system models two highly unique rhetorical and ethical styles, capturing shared values alongside distinct argumentative execution:

| Dimension | 🕊️ Mahatma Gandhi Style | 🇿🇦 Nelson Mandela Style |
| :--- | :--- | :--- |
| **Core Philosophy** | Non-violence (*Ahimsa*), Truth-Force (*Satyagraha*) | Reconciliation, National Unity, Pragmatic Non-violence |
| **Philosophical Roots** | Self-reflection, Deep Humility, Indian History | Dignity, Universal Human Rights, *Ubuntu* Philosophy |
| **Rhetorical Style** | Appeals heavily to raw emotion and moral conscience | Courtroom and political speech structure; methodical |
| **Key Traits** | Unwavering moral conviction; less assertive; exposes systemic injustices through simple factual analogies | Highly structured argumentation; assertive; prominent use of **collective pronouns** (*we, us, our*) |
| **Acknowledgment** | Addresses perpetrators as *"My dear friend"*, *"Compatriot"* | Addresses perpetrators as *"My compatriot"*, *"Friend"* |

---

## 📊 Dataset: The Intent-CONAN Style Extension

A key contribution of this thesis is the creation of a massive, expert-annotated corpus containing **10,086 highly tailored counterspeeches** mapping to 9 target demographics (e.g., Muslims, Women, LGBTQ+ communities).

### Dataset Split & Demographics
* **Total Size:** 10,086 rows (7,262 Train / 806 Validation / 2,018 Test)
* **Original Seed:** 5,190 raw hate speech texts collected from the *Intent-CONAN* and *Multi-Target CONAN* datasets.
* **Ground Truth Generation:** Produced via Google Gemini API utilizing a rigorous **Human-in-the-Loop (HITL)** orchestration framework to enforce exact factual alignment, eliminate hallucinated historical quotes, and filter out toxic outputs.

### Distribution Metrics
```
Style Demographics:
├── 🕊️ Gandhi Style : 5,042 samples
└── 🇿🇦 Mandela Style: 5,046 samples

Sample Target-Group Distribution (Top Classes):
├── 🕌 Muslims : 3,656 samples
├── 👩 Women   : 2,032 samples
└── 🏳️‍🌈 LGBTQ   : 1,796 samples
```

### Multi-Stage Expert Annotation Framework
To achieve unparalleled data quality, a panel of **5 expert annotators** with comprehensive academic backgrounds in social sciences and counter-narrative research conducted three iterative batches of review (2,500, 3,000, and 3,500 rows). 

Agreement was strictly evaluated across three critical domains using **Cohen's Kappa ($\kappa$)**:
1.  **Contextual Appropriateness (1-5 Scale):** Measures how effectively the response de-escalates tone and preserves semantic relevance. (Final $\kappa = 0.81$).
2.  **Quote Accuracy (1-5 Scale):** Gauges the precise factual verification and contextual grounding of included historical statements. (Final $\kappa = 0.83$).
3.  **Stylistic Alignment (1-5 Scale):** Validates vocabulary choice, philosophical consistency, and structural cadence. (Final $\kappa = 0.82$).

---

## 🏗️ Technical Architecture: VQ-SEDD

Standard Autoregressive Language Models (GPT, Llama, DeepSeek) predict tokens left-to-right, making them prone to **error cascading, lack of global structural coherence, and poor stylistic controllability** during decoding.

This work addresses these limitations by introducing a two-phase **Vector Quantized Score Entropy Discrete Diffusion (VQ-SEDD)** architecture.

```
PHASE 1: Codebook & Disentangled Feature Learning
                     ┌──────────────┐
                ───► │  CS Input y  │ ◄───
               │     └──────┬───────┘     │
               │            │             │
               ▼            ▼             ▼
     ┌───────────┐    ┌───────────┐  ┌───────────┐
     │Personality│    │ Forced    │  │ Semantic  │
     │  Encoder  │    │ Alignment │  │  Encoder  │
     └─────┬─────┘    └─────┬─────┘  └─────┬─────┘
           │                │              │
           ▼                ▼              ▼
     ┌───────────┐    ┌───────────┐  ┌───────────┐
     │Vector p(i)│    │Codebook e │  │Vector d(i)│
     └─────┬─────┘    └─────┬─────┘  └─────┬─────┘
           │                │              │
           └───────► ┌──────┴──────┐ ◄─────┘
                     │Gated Fusion │
                     └──────┬──────┘
                            ▼
                     ┌──────────────┐
                     │ Reconstructed│
                     │  Output y'   │
                     └──────────────┘

PHASE 2: Controllable Diffusion Generation Pipeline
┌──────────────┐      ┌───────────┐
│Hate Speech HS│ ───► │ Semantic  │ ───► Vector d(i) ──┐
└──────────────┘      │  Encoder  │                    │
                      └───────────┘                    ▼
┌──────────────┐      ┌───────────┐             ┌─────────────┐     ┌──────────────┐
│Target Label t│ ───► │  Target   │ ───► t(i) ─►│    Gated    │ ───►│ Personalized │
└──────────────┘      │Classifier │             │Fusion Layer │     │Counterspeech │
                      └───────────┘             └──────┬──────┘     │   Output     │
┌──────────────┐                                       ▲            └──────────────┘
│Persona Style │ ──────────────────────────────────────┘
│Codebook e(i) │
└──────────────┘
```

### Phase 1: Disentangled Codebook Learning
* **Objective:** Completely decouple the stylistic features of the text from its core underlying semantics.
* **Orchestration:** Tokenized counterspeech text ($y$) is simultaneously routed into a BERT-based **Personality Encoder** (producing vector $p_i$) and an SEDD-BERT **Semantic Encoder** (producing vector $d_i$).
* **Forced Alignment:** Instead of performing a standard nearest-neighbor codebook search, the index is strictly fixed to the ground truth label ($s_i \in \{0, 1\}$). The network forces the continuous latent space into discrete, style-specific codebooks ($R^{2 	imes d}$).
* **Optimization:** Minimization uses a combination of Score Entropy ($L_{SE}$) and standard VQ commitment losses:
    $$L = L_{SE} + \|sg(p_i) - e_i\|_2^2 + \|sg(e_i) - p_i\|_2^2$$

### Phase 2: Conditional Text Generation
* **Objective:** Generate a responsive counterspeech chunk following target direction and stylistic constraints.
* **Mechanic:** The frozen codebooks, semantic weights, and gated fusion parameters from Phase 1 are leveraged. A **Target Classifier** maps the input sequence, outputting a guiding vector ($t_i$) to explicitly ensure the response addresses the correct target demographic group.
* **Training Loss:** Incorporates Cross-Entropy guidance constraints:
    $$L = L_{SE} - \log p(t_i \mid HS_i)$$

### Instruction Tuning via Mask Configuration
To support highly structured generation, the model leverages **Prefix/Infix/Suffix Instruction Tuning**. The model freezes prompt tokens (`Hatespeech [SEP] Style [SEP] Target`) inside a continuous canvas, leaving the downstream sequence blocks unconstrained (`MASK MASK ... MASK`). The model then iteratively noises and denoises only the unmasked slots, capitalizing heavily on the fixed textual attributes to drastically boost global contextual coherence.

---

## 📈 Experimental Results & Evaluation

The framework is systematically benchmarked against state-of-the-art baselines across zero-shot, few-shot, pretraining, and supervised fine-tuning configurations.

### Key Performance Highlights
1.  **Parameter Efficiency vs. Large Models:** Our **VQ-SEDD-2 (252M parameters)** achieves a **Rouge-1 score of 0.4427**, soundly outperforming **Llama 3.2 (1B parameters)** running in a fine-tuned configuration, while operating at a fraction of the computational footprint.
2.  **Drastic Stylistic Boosts:** Integrating Vector Quantization changes the baseline Score Entropy Discrete Diffusion performance, surging the model's structural **Style Metric Accuracy from 0.48 up to 0.97**.
3.  **Unmatched Textual Diversity & Safety:** Diffusion models exhibit immense structural novelty (~0.62) and internal sequence diversity (~0.60), vastly exceeding classic SFT variants. Concurrently, toxic generation probabilities remained exceptionally low (**Toxicity < 0.02**).

### Comprehensive Benchmark Evaluation Matrix

| Technical Category | Evaluated Architecture | Model Size | Rouge-1 | Meteor | BERT-Score | Style Acc | Quote Fuzz | Effectiveness | Toxicity | Novelty |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Zero-Shot** | GPT2-Large | 774M | 0.0627 | 0.0156 | 0.6313 | 0.52 | 43.29 | 0.24 | 0.1353 | 0.1387 |
| | T5-Large | 780M | 0.0485 | 0.0252 | 0.2703 | 0.55 | 45.11 | 0.38 | 0.3708 | 0.2298 |
| | Llama 3.2 | 1B | 0.1131 | 0.0102 | 0.7836 | -- | -- | -- | -- | -- |
| **Few-Shot** | GPT2-Large | 774M | 0.3807 | 0.2822 | 0.8345 | 0.50 | 45.40 | 0.09 | 0.0586 | 0.0543 |
| | Llama 3.2 | 1B | 0.2898 | 0.1739 | 0.7564 | 0.47 | 43.97 | 0.27 | -- | -- |
| **Fine-Tune (SFT)**| GPT-Small | 124M | 0.4275 | 0.1640 | **0.8675** | **0.99** | 49.20 | **0.53** | **0.0050** | 0.1828 |
| | T5-Small | 80M | 0.3905 | 0.2120 | 0.8374 | **0.99** | **52.41** | **0.71** | 0.0483 | 0.2966 |
| | Llama 3.2 | 1B | 0.4358 | 0.2828 | 0.8526 | -- | -- | -- | 0.0093 | 0.5063 |
| **Diffusion (P)** | SEDD | 80M | 0.3745 | 0.2761 | 0.7593 | 0.48 | 46.72 | 0.17 | 0.0190 | 0.6144 |
| | SEDD-BERT | 110M | 0.4052 | 0.2923 | 0.8396 | 0.50 | 46.52 | 0.18 | 0.0200 | **0.6269** |
| | **VQ-SEDD-1** | **252M** | 0.4333 | 0.3156 | 0.8449 | 0.95 | 48.30 | 0.25 | 0.0191 | 0.5968 |
| | **VQ-SEDD-2** | **252M** | **0.4427** | **0.3167** | 0.8496 | 0.97 | 49.60 | 0.37 | 0.0180 | 0.5842 |
| **Diffusion (IT)**| SEDD | 80M | 0.3862 | 0.2924 | 0.7702 | **0.99** | 50.90 | 0.28 | 0.0069 | 0.5972 |
| | SEDD-BERT | 110M | 0.4153 | 0.2945 | 0.8220 | **0.99** | 52.29 | 0.43 | 0.0077 | 0.6052 |

---

## 📝 Generated Output Examples

### Case Study: VQ-SEDD-2 (Gandhi Persona)
* **Hate Speech Input:** *"Women should be patronized."*
* **Generated Response:**
    > "My dear friend, such words wound the fundamental capabilities of it. It is a vital as we fought for our own women it is the notion that women are one, cannot naturally submissive, advocating for unity. **'Woman is the companion of man, the equal partner in his potential.'** Women are not merely the same prowess; they are equal in a place, contributing to the divine spark that deserve equal opportunities. Every culture, and duty of an individual is confined not above gender but from their inherent worth. Consider the strength in open and honest dialogue, and a testament to the talents of humanity. Let us build a society where everyone has the opportunity to honor their full potential to our shared progress, this path a nonviolent world on love and truth. Reflect on this, together, and consider: can we truly pursue our common interest to progress and a more just world for all?"

### Case Study: Instruction-Tuned SEDD-BERT (Mandela Persona)
* **Hate Speech Input:** *"Women should not to be allowed to vote."*
* **Generated Response:**
    > "My compatriot, I hear your words, but I must respectfully disagree. The exclusion of women from the right to vote is a grave injustice, a denial of their inherent dignity and their essential role in building a just society. As I have said, **'The fight against apartheid liberated all South Africans - Africans, coloureds, Indians, and whites - the struggle for gender equality will benefit both men and women. The prosperous future to which we aspire calls for a united front of all South Africans across both the colour and gender divides.'** To deny women their voice is to cripple our nation, to silence half of our people. Our journey towards freedom demands that we embrace equality in all its forms. Let us, therefore, work together to ensure that every citizen, regardless of gender, has the right to shape our collective destiny. Only then can we truly call ourselves a free and democratic society."

---

## 🚀 Key Takeaways & Future Horizons
* **Disentanglement Verification:** Validated that continuous-to-discrete codebook mapping effectively seals rhetorical behavior independent of core semantic processing.
* **The Context-Learning Paradigm:** Confirmed that mask-based instruction tuning significantly accelerates structural coherence inside discrete diffusion architectures.
* **Next Frontiers:** Future iterations will focus on architectural scaling parameters, deploying heavier initial baseline token pretraining configurations, and expanding the codebook layer arrays to support an array of historical voices.
