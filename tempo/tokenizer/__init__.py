"""Time series tokenizers for TEMPO.

Converts raw time series into discrete token sequences that an LLM can process.
All tokenizers share the same interface:
    tokenize(signal) -> Tensor[int]   (encode to discrete codes)
    decode(codes) -> Tensor            (reconstruct signal)

Available tokenizers:
    - TOTEM: VQ-VAE with learned 256-entry codebook, 4:1 compression.
        Trained on generic time series (UCR archive).
        Located at tempo/tokenizer/totem.py

    - FSQ: Finite Scalar Quantization with CNN encoder.
        Grid quantization (no learned codebook), configurable levels.
        Located at tempo/tokenizer/fsq.py

    - FSQ Transformer: FSQ with Transformer encoder for global context.
        Each code position sees the entire signal via self-attention before
        quantization. Produces structured code sequences (4-10% self-transitions)
        vs CNN FSQ (~0.2%). Inspired by Archetype.
        Located at tempo/tokenizer/fsq_transformer.py

Training:
    python -m tempo.tokenizer.train_fsq_transformer --levels 5 5 5 5 --epochs 100
"""

from .fsq import FSQTokenizer, FSQConfig
from .fsq_transformer import FSQTransformerTokenizer, FSQTransformerConfig
from .totem import TOTEMTokenizer

__all__ = [
    "TOTEMTokenizer",
    "FSQTokenizer", "FSQConfig",
    "FSQTransformerTokenizer", "FSQTransformerConfig",
]
