"""Dual-mode proposition extraction: NLP (rule + model) or LLM."""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Optional

from src.schemas.models import Proposition
from src.utils.llm_client import BaseLLMClient, format_messages

logger = logging.getLogger(__name__)


class PropositionExtractor:

    def __init__(
        self,
        mode: str = "llm",
        llm_client: Optional[BaseLLMClient] = None,
        nlp_inference_model_path: str = "",
        language: str = "en",
        device: str = "cpu",
    ):
        self.mode = mode
        self.language = language
        self.llm_client = llm_client
        self._rule_extractor: Optional[RuleBasedExtractor] = None
        self._model_extractor: Optional[ModelBasedExtractor] = None

        if mode == "nlp":
            self._rule_extractor = RuleBasedExtractor(language=language)
            if nlp_inference_model_path:
                self._model_extractor = ModelBasedExtractor(
                    model_path=nlp_inference_model_path, device=device
                )

    async def extract(self, session_text: str, session_id: str = "") -> list[Proposition]:
        if self.mode == "nlp":
            return self._extract_nlp(session_text, session_id)
        else:
            return await self._extract_llm(session_text, session_id)

    def _extract_nlp(self, session_text: str, session_id: str) -> list[Proposition]:
        rule_props = self._rule_extractor.extract(session_text, session_id)
        logger.info(f"Rule-based extraction: {len(rule_props)} propositions")

        model_props = []
        if self._model_extractor:
            model_props = self._model_extractor.extract(session_text, session_id)
            logger.info(f"Model-based extraction: {len(model_props)} propositions")

        merged = self._merge_deduplicate(rule_props, model_props)
        logger.info(f"Merged propositions: {len(merged)}")
        return merged

    async def _extract_llm(self, session_text: str, session_id: str) -> list[Proposition]:
        prompt = self._build_extraction_prompt(session_text)
        messages = format_messages(user_message=prompt)
        response = await self.llm_client.chat(messages, temperature=0.3)
        raw_props = self._parse_llm_response(response.content)

        propositions = []
        for i, item in enumerate(raw_props):
            prop = Proposition(
                prop_id=f"P{i+1}",
                text=item.get("text", item.get("fact", "")),
                source="llm",
                session_id=session_id,
                time=item.get("time", ""),
                subject=item.get("subject", ""),
                predicate=item.get("predicate", ""),
                object_value=item.get("object_value", item.get("entities_or_values", "")),
                entities=item.get("entities", []),
            )
            propositions.append(prop)

        logger.info(f"LLM extraction: {len(propositions)} propositions")
        return propositions

    def _build_extraction_prompt(self, session_text: str) -> str:
        if self.language == "zh":
            return PROPOSITION_EXTRACTION_PROMPT_ZH.format(session_text=session_text)
        return PROPOSITION_EXTRACTION_PROMPT_EN.format(session_text=session_text)

    def _parse_llm_response(self, content: str) -> list[dict]:
        content = content.strip()

        parsed = self._try_parse_json(content)
        if parsed is not None:
            return parsed

        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
        if match:
            parsed = self._try_parse_json(match.group(1))
            if parsed is not None:
                return parsed

        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            parsed = self._try_parse_json(match.group(0))
            if parsed is not None:
                return parsed

        logger.warning("Failed to parse LLM proposition response")
        return []

    def _try_parse_json(self, text: str) -> list[dict] | None:
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [item for item in result if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass
        return None

    def _merge_deduplicate(
        self, rule_props: list[Proposition], model_props: list[Proposition]
    ) -> list[Proposition]:
        merged = list(rule_props)
        existing_texts = {p.text.lower().strip() for p in rule_props}

        for prop in model_props:
            normalized = prop.text.lower().strip()
            if normalized not in existing_texts:
                merged.append(prop)
                existing_texts.add(normalized)

        for i, prop in enumerate(merged):
            prop.prop_id = f"P{i+1}"
        return merged


class RuleBasedExtractor:

    def __init__(self, language: str = "en"):
        self.language = language
        self._number_pattern = re.compile(
            r'(\d+[\.\d]*)\s*(%|mg|kg|ml|mmHg|bpm|℃|°C|cm|mm|g/dL|mmol/L|mg/dL)',
            re.IGNORECASE,
        )
        self._date_pattern = re.compile(
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|'
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{4})',
            re.IGNORECASE,
        )
        self._zh_date_pattern = re.compile(
            r'(\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日|\d+天前|\d+周前|上周|本周|昨天|今天|明天)'
        )

    def extract(self, session_text: str, session_id: str) -> list[Proposition]:
        propositions = []
        sentences = self._split_sentences(session_text)

        for sent in sentences:
            numbers = self._number_pattern.findall(sent)
            dates = self._date_pattern.findall(sent) + self._zh_date_pattern.findall(sent)

            if numbers or dates:
                prop = Proposition(
                    prop_id="",
                    text=sent.strip(),
                    source="rule",
                    session_id=session_id,
                    time=dates[0] if dates else "",
                    entities=[f"{v}{u}" for v, u in numbers] if numbers else [],
                )
                propositions.append(prop)

        return propositions

    def _split_sentences(self, text: str) -> list[str]:
        if self.language == "zh":
            parts = re.split(r'(?<=[。！？\n])', text)
        else:
            parts = re.split(r'(?<=[.!?\n])\s+', text)
        return [s.strip() for s in parts if s.strip() and len(s.strip()) > 10]


class ModelBasedExtractor:

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self._model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        logger.info(f"Loading inference model from {self._model_path}")
        self._tokenizer = T5Tokenizer.from_pretrained(self._model_path)
        self._model = T5ForConditionalGeneration.from_pretrained(self._model_path)
        self._model.to(self.device)
        self._model.eval()
        logger.info("Inference model loaded")

    def extract(self, session_text: str, session_id: str) -> list[Proposition]:
        self._load_model()
        import torch

        sentences = re.split(r'(?<=[.!?。！？\n])\s*', session_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]

        propositions = []
        for sent in sentences:
            prompt = f"Extract the key factual proposition from this sentence: {sent}"
            inputs = self._tokenizer(
                prompt, return_tensors="pt", max_length=512, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs, max_new_tokens=128, num_beams=2, early_stopping=True
                )
            decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            if decoded and len(decoded) > 10 and decoded.lower() != sent.lower():
                prop = Proposition(
                    prop_id="",
                    text=decoded,
                    source="model",
                    session_id=session_id,
                )
                propositions.append(prop)

        return propositions


PROPOSITION_EXTRACTION_PROMPT_ZH = """你是一个精确的命题提取助手。给定一段对话记录，从中提取所有原子化命题（Propositions）。

## 命题格式
每个命题必须包含以下字段：
- **text**: 完整的命题陈述（一个独立的、可验证的事实）
- **time**: 事件发生的时间（若有）
- **subject**: 命题的主体
- **predicate**: 谓词/动作/关系
- **object_value**: 宾语/数值/结果
- **entities**: 涉及的关键实体列表

## 提取要求
1. 每个命题必须是原子化的，只包含一个独立事实
2. 命题必须具体、可验证，包含具体数值、时间、实体名称
3. 全面覆盖：事实陈述、时间信息、决策行动、状态变化、因果关系、观点偏好
4. 不遗漏任何关键信息

## 会话内容
{session_text}

## 输出
以 JSON 数组形式输出所有命题：
```json
[
  {{"text": "...", "time": "...", "subject": "...", "predicate": "...", "object_value": "...", "entities": ["...", "..."]}},
  ...
]
```"""


PROPOSITION_EXTRACTION_PROMPT_EN = """You are a precise proposition extraction assistant. Given a conversation record, extract ALL atomic propositions from it.

## Proposition Format
Each proposition MUST contain:
- **text**: Complete propositional statement (one independent, verifiable fact)
- **time**: When the event occurred (if available)
- **subject**: The subject of the proposition
- **predicate**: The verb/action/relationship
- **object_value**: The object/value/result
- **entities**: List of key entities involved

## Extraction Requirements
1. Each proposition must be atomic — containing exactly ONE independent fact
2. Propositions must be specific and verifiable with concrete values, times, entity names
3. Cover all dimensions: facts, temporal info, decisions, state changes, causal relations, opinions
4. Do NOT omit any key information

## Session Content
{session_text}

## Output
Output all propositions as a JSON array:
```json
[
  {{"text": "...", "time": "...", "subject": "...", "predicate": "...", "object_value": "...", "entities": ["...", "..."]}},
  ...
]
```"""
