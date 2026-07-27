# TEMPO: Time Series Understanding via Discrete Tokenization


[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

TEMPO is a backbone-agnostic framework that turns any decoder-only LLM into a time-series reasoner. Signals are encoded into discrete tokens by a small FSQ-Transformer quantizer, aligned with the LLM's embedding space in a Phase 0 pass, and trained to answer questions and classify patterns in a Phase 1 pass.

![TEMPO Architecture](assets/architecture.svg)

> [!WARNING]
> **This repository is not yet functional.** Model checkpoints and trained weights are not publicly released — you cannot run inference or reproduce results without them. Full release coming soon.

## Architecture

Three components, trained in sequence:

1. **Tokenizer** (1.6M params, frozen after pre-training). Maps normalized real-valued samples to discrete codes via FSQ-Transformer with `levels = [5,5,5,5]` (625 codes, 4:1 compression).

2. **Phase 0: Alignment.** LLM frozen; a small projection is trained so that tokenizer code embeddings align with the LLM's input space.

3. **Phase 1: Instruction tuning.** LoRA/DoRA adapters trained on downstream tasks (MCQ, captioning, CoT, classification) with signal tokens inlined into the chat template.

## Install

```bash
git clone https://github.com/Forgis-Labs/TEMPO.git
cd TEMPO
uv pip install -e .
```

Python >= 3.10. Dependencies listed in [pyproject.toml](pyproject.toml).

## Quick Start [WIP]

### Inference

```python
import tempo

model = tempo.TEMPO.from_pretrained(
    "checkpoints/phase1_best.pt",
    fsq_ckpt="checkpoints/fsq_transformer_rope_625_best.pt",
    llm_id="Qwen/Qwen3-4B",
    use_dora=True,
)

answer = model.analyze(signal, question="Describe the trend in this signal.")
```

## Training

```python
from tempo import TEMPO, TEMPOConfig
from tempo.train import run_pipeline, PipelineConfig

model = TEMPO(TEMPOConfig(llm_id="Qwen/Qwen3-4B", lora_r=32, use_dora=True))
run_pipeline(model, PipelineConfig(batch_size=4, grad_accum=4))
```

## Evaluation

```python
from tempo.eval import evaluate

result = evaluate(model, test_dataset, output_dir="results/phase1",
                  dataset_name="har", task="classification")
print(result.accuracy, result.f1_macro)
```

## Citation

```bibtex
@misc{2026tempo,
  title  = {TEMPO: Time Series Understanding via Discrete Tokenization},
  year   = {2026},
  url    = {https://github.com/Forgis-Labs/TEMPO},
}
```

## License

[CC BY-NC-SA 4.0](LICENSE) — free for research and non-commercial use.
