"""Tokenizer wrapper around llama-cpp-python's native tokenizer."""

from typing import List, Optional


class Tokenizer:
    """Wraps llama-cpp-python's tokenization capabilities."""

    def __init__(self, llama_instance):
        self._llama = llama_instance

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        """Convert text to token IDs."""
        return self._llama.tokenize(
            text.encode("utf-8"), add_bos=add_bos, special=True
        )

    def decode(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        return self._llama.detokenize(tokens).decode("utf-8", errors="replace")

    def decode_single(self, token: int) -> str:
        """Decode a single token ID to text."""
        return self._llama.detokenize([token]).decode("utf-8", errors="replace")

    @property
    def bos_token_id(self) -> int:
        return self._llama.token_bos()

    @property
    def eos_token_id(self) -> int:
        return self._llama.token_eos()

    @property
    def vocab_size(self) -> int:
        return self._llama.n_vocab()
