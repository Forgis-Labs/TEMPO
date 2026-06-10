"""TEMPO — Time Series Understanding via Discrete Tokenization.

A backbone-agnostic framework that turns any decoder-only LLM into a
time-series reasoner through discrete FSQ-Transformer tokenization.

Quick start::

    import tempo

    # Load a pretrained model
    model = tempo.TEMPO.from_pretrained(
        "checkpoints/phase1_best.pt",
        fsq_ckpt="checkpoints/fsq_transformer_625_best.pt",
        llm_id="Qwen/Qwen3-4B",
    )

    # Analyze a signal
    result = model.analyze(signal, question="What is the trend?")
    print(result)

Submodules:
    tempo.model       Model + tokenizer adapters
    tempo.tokenizer   FSQ / FSQ-Transformer / TOTEM tokenizers
    tempo.data        Dataset builders
    tempo.train       Training pipeline (Phase 0 + Phase 1, curriculum)
    tempo.eval        Evaluation, sensitivity, baseline scoring
    tempo.analysis    Signal-domain analysis (ECG, sleep, walking)
"""

__version__ = "0.2.0"

# Lazy imports: avoids pulling in peft/transformers when only
# the tokenizer subpackage is needed.
def __getattr__(name):
    if name == "TEMPO" or name == "TEMPOConfig":
        from tempo.model.tempo import TEMPO, TEMPOConfig
        return TEMPO if name == "TEMPO" else TEMPOConfig
    raise AttributeError(f"module 'tempo' has no attribute {name!r}")
