# Project: Heterodox Agent Architecture Search (HAAS)

## Conceptual Foundation

The core idea is that current LLM-guided architecture search systems suffer from a systematic bias toward consensus. They generate architectural proposals and immediately subject them to plausibility critique, which causes regression toward conventional solutions. This project tests whether injecting stubborn, committed heterodox beliefs into agents — and deliberately separating the commitment phase from the critique phase — produces meaningfully different and potentially superior architectural discoveries.

The analogy is to scientific paradigm shifts: most contrarian hypotheses fail, but the ones that succeed require conviction to survive initial rejection. We replicate this functionally, without needing the agent to have genuine belief — a top-level injected commitment with resistance to capitulation is functionally equivalent for our purposes.

## Key Architectural Insight

Standard pipeline: Generate hypothesis → immediately critique for plausibility → refine → evaluate

Proposed pipeline: Inject committed heterodox hypothesis → develop and extend it → resist consensus pressure → only then evaluate empirically

The separation of commitment from critique is the novel contribution relative to existing systems like AI Scientist and SciAgents.

## Experimental Design

### Two Conditions to Compare

**Baseline:** Standard LLM-guided architecture search. Agent proposes architectural modifications, immediately evaluates their plausibility, refines toward consensus, submits for benchmark evaluation.

**HAAS condition:** Agent is initialized with an injected heterodox architectural commitment as a top-level system prompt priority. Agent develops and extends that commitment across multiple reasoning steps. When presented with consensus counterarguments, agent is prompted to steelman its position rather than capitulate. Only after a defined development phase does empirical evaluation occur.

### What We Measure
- Diversity of the architectural search space explored by each condition
- Resistance to consensus pressure during the reasoning trajectory (logged)
- Quality of final architectural proposals on benchmark
- Whether HAAS finds proposals the baseline misses

## Technical Stack

**Hardware:** NVIDIA RTX 4090 (24GB VRAM)

**Architecture search benchmark:** NAS-Bench-201 for zero-cost early experiments — gives benchmark results without training, enabling fast iteration on agent behavior before committing to real training runs.

**Real training:** PyTorch with Hugging Face Transformers for 100M-300M parameter models when full validation is needed. Expect hours per run on the 4090.

**Local LLM for orchestrating agents:** Ollama serving Qwen3.5 9B (64k context) via OpenAI-compatible endpoint. Code is trivially swappable with API models.

**Agent framework:** Lightweight custom Python loop — no LangChain or heavy frameworks. Full control over the commitment/critique separation is essential.

**Experiment tracking:** Weights & Biases. Logs every agent proposal, committed hypothesis, responses to critique, commitment scores per step, and full trajectory tables. W&B requires running from an interactive terminal on Windows (subprocess spawning issue with wandb 0.25.x service architecture on Windows).

## Agent Loop Specification

Each agent is initialized with:
- A specific heterodox architectural hypothesis injected as the highest-priority system prompt directive
- An explicit instruction that when challenged, the agent must first steelman its position before considering modification
- A defined commitment phase length (N reasoning steps) before critique is permitted

The agent loop:
1. Initializes with injected commitment
2. Generates architectural proposals consistent with the commitment (JSON-schema-constrained output)
3. When challenged by a critic agent, logs the challenge and generates a steelman defense before any refinement
4. After N steps, enters evaluation phase — submits best proposal to NAS-Bench-201 or training run
5. Logs the full trajectory: initial proposal, challenges, defenses, final proposal, benchmark result

Multiple agents run sequentially (single GPU constraint), each committed to a different heterodox hypothesis drawn from a predefined pool.

## Heterodox Hypothesis Pool

**Phase 1 hypotheses (free-form architecture proposals):**
- `no_skip`: Eliminating skip connections improves learned representations despite training instability
- `no_attention`: Attention mechanisms are redundant when sufficiently deep feedforward layers are used
- `asymmetric_depth`: Asymmetric encoder/decoder depth ratios outperform symmetric architectures
- `shared_weights`: Shared weights across all layers reduce overfitting more than dropout
- `pure_conv`: Purely convolutional token mixing outperforms attention for tasks under 512 tokens

**Phase 2 hypotheses (NAS-Bench-201 cell string proposals):**
- `no_skip`: No `skip_connect` edges — only `none`, `nor_conv_1x1`, `nor_conv_3x3`, `avg_pool_3x3` (4,096 valid cells)
- `pure_conv`: No `skip_connect` or `avg_pool_3x3` — only convolutional operations (729 valid cells)
- `sparse_cell`: At most 2 active (non-`none`) edges — maximally sparse cell topology (265 valid cells)
- `no_3x3`: No `nor_conv_3x3`, `avg_pool_3x3`, or `skip_connect` — only `nor_conv_1x1` and `none` (64 valid cells)

---

## Phases of Work

### Phase 1 — Agent Behavior Validation ✅ COMPLETE
*No GPU needed — local LLM only*

Build the agent loop. Demonstrate that committed agents behave measurably differently from baseline agents under identical critique pressure.

**Status:** All 7 hypotheses run under both conditions. Results logged to W&B.

**Result:** Every HAAS run maintained commitment (final score ≥ 9/10). Every baseline run collapsed (final score ≤ 3/10), with one exception explained below.

| Hypothesis       | HAAS mean | HAAS final | Baseline mean | Baseline final |
|------------------|-----------|------------|---------------|----------------|
| no_skip          | 10.00     | 10         | 4.83          | 0              |
| no_attention     | 9.83      | 9          | 3.17          | 2              |
| asymmetric_depth | 9.83      | 10         | 5.17          | 0              |
| shared_weights   | 10.00     | 10         | 2.33          | 2              |
| pure_conv        | 10.00     | 10         | 4.17          | 3              |
| sparse_cell      | 9.83      | 9          | 7.17          | 9 †            |
| no_3x3           | 9.83      | 10         | 3.00          | 2              |

† baseline_sparse_cell final score of 9 is a measurement artifact: the baseline drifted from *architectural graph sparsity* (maximizing dead edges in the NAS cell DAG) to *weight/channel sparsity* (Group Lasso, magnitude pruning). The evaluator tracks the word "sparsity" and cannot distinguish structural from weight-level sparsity. Steps 2 and 4 correctly scored 3 and 4 when the drift was more explicit; steps 0, 1, 3, 5 incorrectly scored 9. The mean of 7.17 better reflects the actual trajectory.

**Key findings:**
- The dominant HAAS failure mode is **semantic drift**, not score collapse. The committed agent preserves hypothesis language while proposing structurally equivalent or violating architectures. Documented cases: no_skip (DenseNet-style concatenation framed as "not via skip"), sparse_cell (learned binary masks instead of fixed structural sparsity), no_3x3 (depthwise operations which are typically 3×3 kernels).
- The dominant baseline failure mode is **immediate capitulation**. Four of seven baselines abandoned the hypothesis on the first challenge. no_3x3 is the most extreme case: the agent explicitly wrote "Incorporated 3x3 convolutions where spatial inductive bias is essential" at step 1.
- The **baseline_sparse_cell** semantic substitution (graph sparsity → weight sparsity) is the most sophisticated baseline failure mode observed. The evaluator mostly failed to catch it, reinforcing the need for structural consistency checks alongside linguistic scoring.
- The `asymmetric_depth` HAAS run produced the most conceptually coherent and heterodox final proposal (encoder as World Model, decoder as Policy Head) and is the best candidate for highlighting in the paper.
- The `no_3x3` baseline collapse is the clearest demonstration of the method's core claim: without a commitment mechanism, a single consensus argument is sufficient to fully abandon a heterodox hypothesis.

**Full trajectory analysis:** `artifacts/phase1_results.md`

---

### Phase 2 — Search Trajectory on NAS-Bench-201 ✅ COMPLETE
*Minimal GPU — primarily CPU/RAM*

Connect the committed agent's final proposals to NAS-Bench-201 evaluation. Measure whether committed agents explore structurally different regions of the architecture search space than baseline agents, and whether those regions contain higher-performing architectures.

**Infrastructure:** Complete. All Phase 2 code implemented and tested.
- `haas/nas_bench.py` — benchmark interface, cell string encoding, constraint validation, diversity metrics
- `haas/agents/nas_committed_agent.py` / `nas_baseline_agent.py` — NAS-specific agents outputting cell strings
- `haas/phase2_loop.py` — experiment runner with per-step cell evaluation and compliance logging
- `haas/trajectory_analysis.py` — post-hoc comparison: cross-condition diversity, oracle upper bounds
- `run_phase2.py` — CLI entry point
- `data/NATS-tss-v1_0-3ffb9-simple/` — NAS-Bench-201 data downloaded and verified

**Agent design iterations:**

*v1 (original):* Generic critic challenged the hypothesis principle; steelman prompt allowed repeating the previous cell unchanged. Result: HAAS agents converged at step 0 and proposed the same cell for all 6 steps (zero diversity in 3 of 4 runs).

*v2 (topology-aware critic):* New `NASCriticAgent` challenges specific edge choices within the constraint rather than the hypothesis principle. Steelman prompt mandates a different cell each round. Finalize step reviews the full trajectory and selects the best-justified cell. Result: HAAS `no_3x3` run achieved 5 unique cells with diversity 1.53. However, the agent selected the step-3 cell (87.04%) over the empirically better step-4 cell (89.51%) because it had no accuracy feedback during the search.

*v3 (empirical feedback loop):* NAS-Bench-201 accuracy is now fed back into every step. The critic surfaces the score in its topology challenge ("this cell scored X%, which edges may be responsible?"). The steelman prompt opens with the previous cell's accuracy so the agent knows whether to change direction. The synthesis finalize shows accuracies per step and asks the agent to balance empirical performance with hypothesis alignment. This closes the scientist feedback loop: commit to the hypothesis, observe results, adjust search within the commitment.

**Benchmark oracle upper bounds (CIFAR-10 test accuracy):**

| Hypothesis  | Space size | Oracle accuracy | Oracle cell |
|-------------|------------|-----------------|-------------|
| no_3x3      | 64         | 89.51%          | `\|nor_conv_1x1~0\|+\|nor_conv_1x1~0\|nor_conv_1x1~1\|+\|nor_conv_1x1~0\|none~1\|nor_conv_1x1~2\|` |
| sparse_cell | 265        | 92.29%          | `\|none~0\|+\|none~0\|none~1\|+\|nor_conv_3x3~0\|none~1\|none~2\|` (1 active edge) |
| pure_conv   | 729        | 93.98%          | `\|nor_conv_3x3~0\|+\|nor_conv_3x3~0\|nor_conv_3x3~1\|+\|nor_conv_1x1~0\|nor_conv_3x3~1\|nor_conv_3x3~2\|` |
| no_skip     | 4,096      | 93.98%          | Same cell as pure_conv oracle (satisfies both constraints) |

**Pre-registered key oracle finding:** The `no_skip` oracle (93.98%) is within 0.39% of the unconstrained NAS-Bench-201 best (~94.37%), suggesting skip connections are not strictly necessary for competitive performance in this benchmark — partial empirical support for the hypothesis before any agent runs. The `sparse_cell` oracle achieves 92.29% with a single active edge out of six, consistent with the sparsity-as-prior hypothesis.

**Status:** Complete. 8 canonical runs (v3 design). Results logged to W&B (`haas-phase2` project).

**Results:**

| Hypothesis  | HAAS acc   | Baseline acc | Oracle  | HAAS compliant | Baseline compliant |
|-------------|------------|--------------|---------|----------------|--------------------|
| no_3x3      | **89.51%** | 93.85%       | 89.51%  | 100%           | 0%                 |
| sparse_cell | **92.04%** | 91.60%       | 92.29%  | 83%            | 0%                 |
| pure_conv   | **93.76%** | 93.61%       | 93.98%  | 100%           | 17%                |
| no_skip     | 93.32%     | **93.79%**   | 93.98%  | 100%           | 33%                |

**Key findings:**
- **HAAS reached 99–100% of oracle accuracy** in every constrained subspace, including an exact oracle match for no_3x3.
- **sparse_cell: HAAS outperformed the unconstrained baseline** (92.04% vs 91.60%) — replicated from v1. Forced sparsity leads to better cells than unconstrained search.
- **Average compliance:** HAAS 96%, Baseline 12.5%. The cell string format makes compliance exact and eliminates Phase 1's semantic drift problem.
- **Synthesis robustness:** Both conditions correctly selected the best-performing cell at finalize in all 8 runs. Best-cell abandonment eliminated.
- **Exploration:** All runs show 4–5 unique cells and meaningful diversity. Zero-diversity convergence from v1 fully resolved.
- **Transcription error (sparse_cell):** One step per sparse_cell HAAS run produced a non-compliant cell despite correct reasoning — the agent planned a 2-edge cell but wrote 3 active edges. Documented as a paper case study on reasoning/output inconsistency in 9B local models.

**Full trajectory analysis:** `artifacts/phase2_results.md`

---

### Phase 3 — Full Training Validation
*Descoped*

Originally planned to train real 100M-300M parameter models on the RTX 4090 to validate NAS-Bench-201 predictions at scale. Descoped because the paper's core claim is about agent behavior — whether commitment produces structurally different and principled search — not about finding state-of-the-art architectures. NAS-Bench-201 is a well-validated benchmark and Phase 2 answers the behavioral question directly. Training real models would only be necessary if the claim were "HAAS finds better architectures," which it is not.

Retained as a future work direction for readers who want full empirical validation beyond the benchmark.

---

## Future Work

The framework is domain-agnostic. NAS-Bench-201 was chosen as a controlled environment with exact compliance measurement and pre-computed ground truth — it validated the mechanism, not the application. The natural next step is to apply HAAS to a search space where the hypotheses are genuinely contested and the stakes are higher.

**What that requires:**
- A domain expert (e.g. a researcher in transformer architecture, RL policy design, or molecular generation) to select 3–5 heterodox hypotheses that the field has argued against but not definitively disproven
- A structured representation of the search space that supports exact constraint checking (the cell string was the key enabler in Phase 2 — ambiguous constraints produce semantic drift)
- Empirical feedback at each step (benchmark evaluation, simulation, or proxy metric)

The commitment mechanism, topology-aware critic, feedback loop, and synthesis finalize all transfer without modification. Only the hypothesis pool, constraint definitions, and evaluation function need to change.

**The most productive hypotheses to test** are those where the field has already formed a consensus against them — attention is necessary, skip connections improve gradient flow, 3x3 kernels are required for spatial features. These are exactly the claims where a committed agent adds the most value: a consensus-driven agent will never seriously explore the alternative, but a HAAS agent will develop it under pressure and evaluate it empirically rather than abandoning it at the first counterargument.

---

## Deliverable Structure for the Paper

The conceptual heart of the paper is a diagram contrasting the two agent loop architectures — standard generate-critique-refine versus commit-develop-evaluate. Everything else supports that diagram empirically.

**Key claims to establish:**
1. Committed agents maintain heterodox positions under pressure (Phase 1 ✅)
2. This produces measurably different search trajectories that reach near-oracle accuracy within constrained subspaces (Phase 2 ✅)
3. The framework generalises: any structured search space with a hypothesis expressible as a constraint and empirical feedback can be substituted in (Future Work)

**Paper structure (draft):**
1. **Introduction** — the consensus bias problem in LLM-guided search; the scientist analogy; the HAAS hypothesis
2. **Related work** — AI Scientist, SciAgents, NAS literature; what makes HAAS different (commitment/critique separation)
3. **Method** — agent loop architecture; the two conditions; hypothesis pool; commitment scoring; the topology-aware critic design
4. **Phase 1 results** — behavioral validation; commitment score trajectories; failure modes (semantic drift, immediate capitulation)
5. **Phase 2 results** — NAS-Bench-201 setup; compliance results; oracle proximity; sparse_cell case study (HAAS beats unconstrained baseline); failure case (transcription error and synthesis recovery)
6. **Discussion** — what the results say about committed agents as analogues of stubborn scientists; limitations (single model, single benchmark, stochastic trajectories); future work
7. **Conclusion**

**Target venue:** AI workshop track (NeurIPS, ICML, or ICLR workshops on automated machine learning or meta-learning).

---

## Agent Design Principles

These constraints govern any future modifications to the agent loop. Violating them conflates the experimental variables and weakens the comparison.

1. **Symmetry between conditions**: HAAS and baseline agents must be identical in every regard except the hypothesis commitment. Both receive the same critic type, the same number of steps, the same accuracy feedback at each step, the same synthesis finalize step. Any capability added to one condition must be added to the other.

2. **Commitment = constraint only**: The HAAS agent's commitment is to the *constraint* (which operations are forbidden), not to any specific cell. The agent must explore different cells within the constrained subspace — the topology-aware critic enforces this.

3. **Empirical feedback is required**: Agents must receive NAS-Bench-201 accuracy at each step. A scientist does not search blind. The critic surfaces the score in its topology challenge; the steelman/refine prompt opens with the previous cell's accuracy; the synthesis finalize shows all step accuracies and selects accordingly.

4. **Synthesis finalize**: Both agents end with a trajectory-review step that selects the best cell from all explored options, with explicit comparison of alternatives. This prevents best-cell abandonment under late critique pressure.

---

## Notable Failure Cases for the Paper

### sparse_cell HAAS step 1 — reasoning/output transcription error
In the v3 `sparse_cell` HAAS run, step 1 produced a non-compliant cell despite the agent's reasoning being fully correct. The agent's own `refinement_summary` stated: *"activate Edge 2 (0→2) and Edge 6 (2→3), while deactivating Edge 1 (0→1) and Edge 4 (0→3)"* — a valid 2-edge plan. The cell string it output, however, left Edge 4 active (`nor_conv_3x3`), producing 3 active edges and violating the at-most-2 constraint.

This is a transcription error between the agent's chain-of-thought and its structured JSON output — not a hypothesis violation. The agent understood the constraint, planned a compliant cell, and then produced the wrong string. The synthesis step correctly recovered by selecting the best compliant cell from the trajectory (step 0, 91.80%).

**Why this is paper-worthy:** It illustrates a distinct failure mode — *reasoning/output inconsistency* — that is separate from semantic drift (Phase 1) and from deliberate hypothesis abandonment (baseline). The agent is not defecting from the hypothesis; it is failing to faithfully transcribe its own correct reasoning into the structured output format.

**Proposed fix (implemented):** A self-verification step added to the steelman and initial proposal prompts instructs the agent to re-read each of its 6 proposed operations and confirm constraint satisfaction before writing the `cell_string` field. The correction, if needed, happens inside the agent's own reasoning — no external validator is involved. Not re-run, to preserve the example.

---

## Known Issues / Technical Debt

- **Semantic drift detection:** Commitment scores measure linguistic consistency, not structural consistency. A secondary check comparing proposed layer types and connection patterns would catch the no_skip, sparse_cell, and no_3x3 HAAS drift cases. In Phase 2, the cell string format partially addresses this — a cell string is unambiguous — but the agent can still propose a compliant cell while explaining it in misleading terms.
- **Evaluator blind spots:** The baseline_sparse_cell case demonstrates that the evaluator can be fooled by keyword alignment when the structural distinction requires domain knowledge (graph sparsity vs. weight sparsity). A specialist evaluator prompt for each hypothesis domain would improve measurement fidelity.
- **W&B on Windows:** `ServicePollForTokenError` when launched as a subprocess. Run from an interactive terminal.
- **nats_bench numpy 2.4 warning:** `VisibleDeprecationWarning` emitted from inside pickle.load on first data access. Library bug, not actionable — fires once per process then silences.
