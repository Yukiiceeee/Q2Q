"""Token-level perplexity-based information density compression."""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PerplexityCompressor:

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        threshold: float = 5.0,
    ):
        self.device = device
        self.threshold = threshold
        self._model = None
        self._tokenizer = None
        self._model_path = model_path
        self._number_pattern = re.compile(r'\d')

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading compressor model from {self._model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path, trust_remote_code=True, torch_dtype=torch.float32
        )
        self._model.to(self.device)
        self._model.eval()
        logger.info("Compressor model loaded")

    def compress(self, session_text: str) -> str:
        if not session_text.strip():
            return session_text

        self._load_model()

        sentences = self._split_sentences(session_text)
        if not sentences:
            return session_text

        kept_sentences = []
        for sentence in sentences:
            compressed = self._compress_sentence(sentence)
            if compressed.strip():
                kept_sentences.append(compressed)

        result = " ".join(kept_sentences)
        ratio = len(result) / max(len(session_text), 1)
        logger.info(
            f"Compression: {len(session_text)} -> {len(result)} chars "
            f"(ratio={ratio:.2f}, threshold={self.threshold})"
        )
        return result

    def _compress_sentence(self, sentence: str) -> str:
        tokens = self._tokenizer.encode(sentence, return_tensors="pt").to(self.device)
        if tokens.shape[1] <= 2:
            return sentence

        with torch.no_grad():
            outputs = self._model(tokens, labels=tokens)
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = tokens[:, 1:]

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

        token_perplexities = token_losses.cpu().numpy()
        token_ids = tokens[0].cpu().tolist()
        decoded_tokens = [
            self._tokenizer.decode([tid], skip_special_tokens=False)
            for tid in token_ids
        ]

        kept_tokens = [decoded_tokens[0]]
        for i, (token_text, ppl) in enumerate(
            zip(decoded_tokens[1:], token_perplexities), start=1
        ):
            if ppl >= self.threshold:
                kept_tokens.append(token_text)
            else:
                if self._is_important_token(token_text):
                    kept_tokens.append(token_text)

        result = "".join(kept_tokens)
        result = self._clean_compressed_text(result)
        return result

    def _is_important_token(self, token: str) -> bool:
        stripped = token.strip()
        if not stripped:
            return False
        if self._number_pattern.search(stripped):
            return True
        structural = {".", ",", "!", "?", ":", ";", "\n", "。", "，", "！", "？", "：", "；"}
        if stripped in structural:
            return True
        if stripped[0].isupper() and len(stripped) > 1:
            return True
        return False

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?。！？\n])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def _clean_compressed_text(self, text: str) -> str:
        text = re.sub(r'\s{2,}', ' ', text)
        text = text.strip()
        return text
