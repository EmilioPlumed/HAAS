# HAAS — Heterodox Agent Architecture Search

Code and data for the paper **"Committed to the Hypothesis: LLM Agents as Stubborn Scientists in Neural Architecture Search"**.

## Overview

HAAS studies whether an LLM agent pre-committed to a non-negotiable architectural hypothesis explores differently than a standard generate-critique-refine baseline. Experiments run on [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201) where constraint compliance and performance are exact and pre-evaluated.

**Key result:** Committed agents maintain ~96% constraint compliance vs 12.5% for the baseline, reach 99–100% of oracle accuracy in every constrained subspace, and in one case outperform the unconstrained baseline entirely.

## Repository structure

```
haas/                   Core library
  agents/               Agent implementations (committed, baseline, critic)
  config.py             LLM and experiment configuration
  hypotheses.py         Seven architectural hypotheses
  llm_client.py         OpenAI-compatible LLM client with retry logic
  logging_utils.py      W&B logging and local JSONL backup
  loop.py               Phase 1 agent loop (free-text proposals)
  phase2_loop.py        Phase 2 agent loop (NAS-Bench-201 cell strings)
  nas_bench.py          NAS-Bench-201 / NATS-TSS interface

run_experiment.py       Entry point for Phase 1 runs
run_phase2.py           Entry point for Phase 2 runs

artifacts/
  phase1_results.md     Phase 1 analysis (7 hypotheses, commitment scores)
  phase2_results.md     Phase 2 analysis (4 hypotheses, NAS-Bench-201)
  run-*/                Phase 1 trajectory tables (JSON)

paper/
  tmlr-style-file-main/
    haas_paper.tex      Paper source
    haas_paper.pdf      Compiled paper
    haas.bib            Bibliography
```

## Requirements

```bash
pip install -r requirements.txt
```

You also need [Ollama](https://ollama.com) running locally with `qwen3.5:9b-64k` pulled:

```bash
ollama pull qwen3.5:9b-64k
```

And the NATS-Bench dataset. On first run `run_phase2.py` will attempt to download it automatically via `gdown`. If that fails, download `NATS-tss-v1_0-3ffb9-simple.tar` manually from the [NATS-Bench releases](https://github.com/D-X-Y/NATS-Bench) and place it in `data/`.

## Running Phase 2 (NAS-Bench-201)

```bash
python run_phase2.py \
  --hypothesis no_skip \
  --condition haas \
  --wandb-project haas-phase2
```

`--condition` is either `haas` or `baseline`. `--hypothesis` is one of:
`no_skip`, `sparse_cell`, `pure_conv`, `no_3x3`.

To run without W&B:

```bash
python run_phase2.py --hypothesis sparse_cell --condition haas --no-wandb
```

## Running Phase 1 (free-text behavioral study)

```bash
python run_experiment.py \
  --hypothesis asymmetric_depth \
  --condition haas
```

## Experimental results

Full trajectories and per-step metrics are logged to W&B. A public report with all Phase 2 runs is available at:

> **[W&B Report — add link here once report is created]**

Phase 1 trajectory tables are included in `artifacts/run-*/trajectory.table.json`.

## Paper

The paper source and compiled PDF are in `paper/tmlr-style-file-main/`. To recompile:

```bash
cd paper/tmlr-style-file-main
pdflatex haas_paper.tex
bibtex haas_paper
pdflatex haas_paper.tex
pdflatex haas_paper.tex
```

Requires a LaTeX distribution with the packages listed in `haas_paper.tex` (all available in MiKTeX / TeX Live).
