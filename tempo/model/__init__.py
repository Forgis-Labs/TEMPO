"""TEMPO model + TOTEM tokenizer."""

# Lazy imports — avoids pulling in peft/transformers when only
# the tokenizer subpackage is needed (e.g., tokenizer training on SageMaker).
def __getattr__(name):
    if name in ("TEMPO", "TEMPOConfig"):
        from .tempo import TEMPO, TEMPOConfig
        return TEMPO if name == "TEMPO" else TEMPOConfig
    if name == "TOTEMTokenizer":
        from .totem import TOTEMTokenizer
        return TOTEMTokenizer
    raise AttributeError(f"module 'tempo.model' has no attribute {name!r}")

__all__ = ["TEMPO", "TEMPOConfig", "TOTEMTokenizer"]
