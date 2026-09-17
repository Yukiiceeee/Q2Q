"""Sentence-level perplexity-based information density compression."""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Protected sentence patterns: dates, numbers, proper nouns, quotes
_DATE_PATTERN = re.compile(
    r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|'
    r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}|'
    r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}|'
    r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*|'
    r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|'
    r'(yesterday|today|tomorrow|last\s+(week|month|year)|next\s+(week|month|year)))',
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r'\d+')
_QUOTE_PATTERN = re.compile(r'["“”‘’]')
_PROPER_NOUN_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b')


class PerplexityCompressor:

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        threshold: float = 3.0,
    ):
        self.device = device
        self.threshold = threshold
        self._model = None
        self._tokenizer = None
        self._model_path = model_path

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
            if self._should_keep(sentence):
                kept_sentences.append(sentence)

        result = " ".join(kept_sentences)
        ratio = len(result) / max(len(session_text), 1)
        logger.info(
            f"Sentence compression: {len(session_text)} -> {len(result)} chars "
            f"(ratio={ratio:.2f}, kept {len(kept_sentences)}/{len(sentences)} sentences)"
        )
        return result

    def _should_keep(self, sentence: str) -> bool:
        if self._is_protected(sentence):
            return True
        avg_ppl = self._compute_sentence_perplexity(sentence)
        return avg_ppl >= self.threshold

    def _compute_sentence_perplexity(self, sentence: str) -> float:
        tokens = self._tokenizer.encode(sentence, return_tensors="pt").to(self.device)
        if tokens.shape[1] <= 2:
            return float("inf")

        with torch.no_grad():
            outputs = self._model(tokens, labels=tokens)
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = tokens[:, 1:]

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

        return float(token_losses.mean().item())

    def _is_protected(self, sentence: str) -> bool:
        if _DATE_PATTERN.search(sentence):
            return True
        if _QUOTE_PATTERN.search(sentence) and len(sentence) > 10:
            return True
        if _PROPER_NOUN_PATTERN.search(sentence):
            return True
        digit_count = len(_NUMBER_PATTERN.findall(sentence))
        if digit_count >= 2:
            return True
        return False

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?。！？\n])\s*', text)
        return [s.strip() for s in sentences if s.strip()]
