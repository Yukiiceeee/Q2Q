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

    async def extract(self, session_text: str, session_id: str = "", session_date: str = "") -> list[Proposition]:
        if self.mode == "nlp":
            return self._extract_nlp(session_text, session_id)
        else:
            return await self._extract_llm(session_text, session_id, session_date)

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

    async def _extract_llm(self, session_text: str, session_id: str, session_date: str = "") -> list[Proposition]:
        prompt = self._build_extraction_prompt(session_text, session_date)
        messages = format_messages(user_message=prompt)
        response = await self.llm_client.chat(messages, temperature=0.3)
        raw_props = self._parse_llm_response(response.content)

        propositions = []
        for i, item in enumerate(raw_props):
            time_val = item.get("time", "")
            subject = item.get("subject", "")
            fact = item.get("fact", item.get("text", ""))
            entities_or_values = item.get("entities_or_values", item.get("entities", []))

            if isinstance(entities_or_values, str):
                entities_or_values = [e.strip() for e in entities_or_values.split(",") if e.strip()]

            # Build self-contained natural language text for embedding
            # The fact field should include subject per prompt instructions,
            # but we prepend subject if it's not already in the fact
            if subject and subject.lower() not in fact.lower():
                text = f"{subject} - {fact}"
            else:
                text = fact

            prop = Proposition(
                prop_id=f"P{i+1}",
                text=text,
                source="llm",
                session_id=session_id,
                time=time_val,
                subject=subject,
                predicate="",
                object_value=fact,
                entities=entities_or_values,
            )
            propositions.append(prop)

        logger.info(f"LLM extraction: {len(propositions)} propositions (structured KP format)")
        return propositions

    def _build_extraction_prompt(self, session_text: str, session_date: str = "") -> str:
        if self.language == "zh":
            template = PROPOSITION_EXTRACTION_PROMPT_ZH
        else:
            template = PROPOSITION_EXTRACTION_PROMPT_EN

        if session_date:
            date_context = f"\n\n## Conversation Date Context\nThis conversation took place on: **{session_date}**\nUse this date as the anchor to resolve all relative time references (e.g., 'yesterday' = one day before {session_date}, 'last week' = the week before {session_date}, 'two months ago' = approximately two months before {session_date}).\n"
            return template.format(session_text=session_text) + date_context
        return template.format(session_text=session_text)

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


PROPOSITION_EXTRACTION_PROMPT_ZH = """你是一个精确的知识点提取助手。给定一段对话记录，从中提取所有结构化知识点（Knowledge Points）。

## 知识点格式
每个知识点必须包含以下四个字段：
- **time**: 事件发生的绝对日期。如果某人在 2023年7月6日 的对话中说"我昨天去了博物馆"，则 time 为 "2023-07-05"。无法确定时间时留空字符串。
- **subject**: 知识点的主体（具体人物名、实体名，不要用代词）
- **fact**: 关于主体的**高密度事实陈述**。必须遵循"动作/状态 + 对象 + 全部修饰语（时间/地点/数量/方式/原因/结果）"模式。保留原文中所有具体信息，不做任何概括。
- **entities_or_values**: 该知识点中能**唯一区分此事件**的关键标识符列表。必须包含：所有专有名词（人名、地名、机构名、书名、药品名等）、所有数值指标（含单位）、所有日期。

## 关键时间规则
- time 字段必须是"事件发生时间"，**不是**对话时间戳。
- 必须将所有相对时间引用（昨天、上周、两天前、那个周末等）转换为绝对日期。
- 对话时间戳信息在对话前缀中，利用它来推算绝对时间。
- 仅当句子中完全没有任何时间引用（显式或隐式）时，time 字段才留空。
- **强制规则**：如果句子包含任何相对时间表达，time 字段绝对不能为空。
- 当无法精确定位但可大致推断时，使用"大约[日期]"格式（如"大约2023年6月"）。

### 时间推理链（遵循以下步骤）
1. 从对话时间戳前缀或上下文标题中确定对话日期。
2. 对每个相对时间引用计算：对话日期 + 偏移 = 事件日期。
3. 推算示例：
   - 对话发生在"2023年7月6日" + "昨天" → time: "2023-07-05"
   - 对话发生在"2023年7月6日" + "上周" → time: "大约2023年6月26日-30日"
   - 对话发生在"2023年7月6日" + "两个月前" → time: "大约2023年5月"
   - 对话发生在"2023年7月6日" + "我上大学的时候" → time: ""（过于模糊，无法推算）

## fact 字段信息密度要求（极其重要）
每条 fact 必须达到"看到这一条就能回答相关问题"的完整程度：

**好的 fact（信息完整、自包含）**：
- "张某因左膝关节疼痛加重2周，于市中心人民医院骨科就诊，膝关节X线显示内侧间隙狭窄"
- "李医生将阿莫西林剂量从每日500mg×3次调整为每日750mg×2次，原因是患者出现了胃肠道不适"
- "患者体重从2023年1月的85kg降至2023年6月的78kg，降幅7kg，归因于低碳饮食方案"

**差的 fact（信息丢失、需要上下文）**：
- "去了医院看骨科" ← 缺少人名、医院名、症状、检查结果
- "调整了药物剂量" ← 缺少药名、原剂量、新剂量、调整原因
- "体重下降了" ← 缺少具体数值、时间跨度、原因

## 强制覆盖维度（每一类都不得遗漏）
1. **具体活动/事件**：提到的任何活动，无论多简短或一笔带过
2. **个人状态/属性**：感情状态、职业、年龄、居住情况、健康、教育、身份
3. **时间事件**：含有具体日期或时间引用的事件
4. **决策与计划**：计划要做的事、正在研究/考虑的事
5. **人际关系**：谁认识谁、什么关系、关系变化
6. **观点与偏好**：喜好、厌恶、爱好、兴趣、评价
7. **量化事实**：数字、时长、频率、数量、金额、排名
8. **因果链条**：做决定的原因、事件的影响、动机

## 关键提取规则
- **信息零损耗**：原文提到的每一个具体数字、名称、日期、地点都必须出现在某条 fact 中。
- **宁多勿少**：提取每一个具体事实，无论多细微。
- **一句多提**：一句话可产生多个知识点，不要合并多个不同事实为一条。
- **顺带提及也提取**：某人顺带提到做了某事，也必须独立提取。
- **保持自包含**：每条 fact 必须独立可理解，包含足够的主体和背景信息。用全名不用代词。
- **全面覆盖**：确保对话中的每一个信息点都被至少一条知识点覆盖。
- **禁止概括**：不要用"等"、"相关"、"之类"等模糊词替代具体列举。

## 会话内容
{session_text}

## 对话数据特别注意事项
- 这是人与人之间的对话。对每个说话者提到的每件事都单独提取知识点。
- 对话中提到的每一个具体细节（人名、地名、日期、计划、偏好、健康信息、关系）都必须单独提取。
- 一个对话轮次可以产生 3-10+ 个知识点，不要将多个不同事实合并为一条。
- 始终使用说话者的全名（如"Melanie"、"Caroline"），绝对不要用"她"、"他"、"他们"等代词。
- fact 字段中必须包含主体姓名，确保 fact 自包含可理解（如"Melanie 去了..."而不是"去了..."）。
- 对话中出现的问题，问题主题和回答内容都应分别提取为知识点。

## 输出
以 JSON 数组形式输出所有知识点（通常一段对话应提取 20-60 条）：
```json
[
  {{"time": "2023-07-05", "subject": "Melanie", "fact": "Melanie带两个孩子（Jake和Lily）去了芝加哥自然历史博物馆参观恐龙化石展，在博物馆待了约3小时", "entities_or_values": ["Melanie", "Jake", "Lily", "芝加哥自然历史博物馆", "恐龙化石展", "3小时"]}},
  {{"time": "", "subject": "Caroline", "fact": "Caroline目前是单身状态，上一段恋情在6个月前结束，前男友名叫Daniel", "entities_or_values": ["Caroline", "Daniel", "6个月前"]}},
  {{"time": "2023-07-01", "subject": "我", "fact": "我在GoodLife健身房开始了为期12周的力量训练计划，每周一/三/五去，每次约1小时", "entities_or_values": ["GoodLife健身房", "12周", "力量训练", "每周3次", "1小时"]}},
  {{"time": "2023-07-04", "subject": "Tom", "fact": "Tom被确诊为2型糖尿病，空腹血糖8.2mmol/L，医生开了二甲双胍500mg每日两次", "entities_or_values": ["Tom", "2型糖尿病", "空腹血糖", "8.2mmol/L", "二甲双胍", "500mg", "每日两次"]}}
]
```"""


PROPOSITION_EXTRACTION_PROMPT_EN = """You are a precise knowledge point extraction assistant. Given a conversation record, extract ALL structured knowledge points from it.

## Knowledge Point Format
Each knowledge point MUST contain these four fields:
- **time**: The absolute date when the event occurred. If someone says "Yesterday I went to the museum" in a conversation dated July 6, 2023, time is "July 5, 2023". Leave as empty string if time cannot be determined.
- **subject**: The subject of the knowledge point (use specific person names or entity names, NEVER use pronouns)
- **fact**: A **high-density factual statement** about the subject. Must follow the pattern: "action/state + object + ALL modifiers (time/place/quantity/manner/cause/result)". Retain ALL specific information from the original text without any generalization.
- **entities_or_values**: List of **uniquely distinguishing identifiers** for this event. MUST include: all proper nouns (person names, place names, organization names, book titles, drug names, etc.), all numeric indicators (with units), all dates mentioned.

## Critical Time Rules
- The time field must be the EVENT occurrence time, NOT the conversation timestamp.
- Convert ALL relative time references (yesterday, last week, two days ago, that weekend, etc.) to absolute dates.
- Use the conversation timestamp prefix to calculate absolute dates.
- Leave as empty string ONLY when there is absolutely NO time reference (explicit or implicit) in the sentence.
- MANDATORY: If a sentence contains ANY relative time expression, the time field MUST NOT be empty.
- When the exact date cannot be pinpointed but can be approximated, use "approximately [date]" (e.g., "approximately June 2023").

### Time Inference Chain (follow these steps)
1. Identify the conversation date from the timestamp prefix or context header.
2. For each relative time reference, calculate: conversation_date + offset = event_date.
3. Examples of resolution:
   - Conversation on "July 6, 2023" + "yesterday" → time: "July 5, 2023"
   - Conversation on "July 6, 2023" + "last week" → time: "approximately June 26-30, 2023"
   - Conversation on "July 6, 2023" + "two months ago" → time: "approximately May 2023"
   - Conversation on "July 6, 2023" + "when I was in college" → time: "" (too vague, cannot resolve)

## Fact Field Information Density Requirement (CRITICAL)
Each fact must be complete enough that "reading this single fact alone can answer a question about it":

**GOOD facts (information-complete, self-contained)**:
- "Sarah visited Dr. Martinez at Downtown Orthopedic Clinic for worsening left knee pain of 2 weeks duration, X-ray showed medial joint space narrowing"
- "Dr. Chen changed amoxicillin dosage from 500mg three times daily to 750mg twice daily because the patient developed gastrointestinal discomfort"
- "Patient's weight decreased from 85kg in January 2023 to 78kg in June 2023, a loss of 7kg, attributed to a low-carb diet plan"
- "Emma recommended the book 'Atomic Habits' by James Clear, saying she read it in 3 days and it changed her morning routine"

**BAD facts (information lost, context-dependent)**:
- "went to the doctor for knee pain" ← missing: WHO, which doctor, which clinic, what findings
- "medication dosage was adjusted" ← missing: which drug, old dose, new dose, reason for change
- "lost some weight" ← missing: specific numbers, time span, method
- "recommended a book" ← missing: WHO recommended, book title, author, why

## Mandatory Coverage Dimensions (do NOT skip ANY category)
1. **Concrete Activities/Events**: ANY activity mentioned, however briefly or casually
2. **Personal Status/Attributes**: relationship status, job, age, living situation, health, education, identity
3. **Temporal Events**: events with specific dates or time references
4. **Decisions & Plans**: planned actions, things being researched/considered
5. **Relationships Between People**: who knows whom, relationship type, changes in relationships
6. **Opinions & Preferences**: likes, dislikes, hobbies, interests, evaluations
7. **Quantitative Facts**: numbers, durations, frequencies, amounts, costs, rankings
8. **Causal Chains**: reasons for decisions, effects of events, motivations

## Critical Extraction Rules
- **Zero information loss**: EVERY specific number, name, date, and place mentioned in the original text MUST appear in some fact.
- **Extract MORE rather than fewer**: extract EVERY concrete fact, no matter how minor.
- **One sentence, multiple KPs**: a single sentence can yield multiple knowledge points — do NOT merge distinct facts into one.
- **Casual mentions count**: if someone casually mentions doing something, it MUST be extracted as a separate KP.
- **Self-contained facts**: each `fact` field must be independently understandable with sufficient subject and context. Use full names, not pronouns.
- **Full coverage**: ensure EVERY piece of information in the conversation is covered by at least one knowledge point.
- **No vague language**: NEVER use "etc.", "related", "some", "various" or similar vague terms to replace specific enumeration.

## Session Content
{session_text}

## Special Attention for Conversational Data
- This is a conversation between people. Extract knowledge points for EVERY speaker separately.
- Every specific detail mentioned (names, places, dates, plans, preferences, health info, relationships) must be extracted as a separate knowledge point.
- A single conversation turn can yield 3-10+ knowledge points. Do NOT summarize multiple facts into one KP.
- Always use the person's actual name (e.g., "Melanie", "Caroline"), NEVER use pronouns like "she", "he", "they".
- Include the subject's name in the fact field so the fact is self-contained (e.g., "Melanie visited..." not just "visited...").
- For questions asked in conversation, extract BOTH the question topic AND the answer as separate KPs.

## Output
Output all knowledge points as a JSON array (typically 20-60 per conversation):
```json
[
  {{"time": "July 5, 2023", "subject": "Melanie", "fact": "Melanie took her two children (Jake and Lily) to the Chicago Natural History Museum to see the dinosaur fossil exhibit, spending approximately 3 hours there", "entities_or_values": ["Melanie", "Jake", "Lily", "Chicago Natural History Museum", "dinosaur fossil exhibit", "3 hours"]}},
  {{"time": "", "subject": "Caroline", "fact": "Caroline is currently single, her previous relationship with Daniel ended 6 months ago", "entities_or_values": ["Caroline", "Daniel", "6 months ago"]}},
  {{"time": "July 1, 2023", "subject": "I", "fact": "I started a 12-week strength training program at GoodLife Gym, going Monday/Wednesday/Friday, approximately 1 hour per session", "entities_or_values": ["GoodLife Gym", "12 weeks", "strength training", "3 times per week", "1 hour"]}},
  {{"time": "July 4, 2023", "subject": "Tom", "fact": "Tom was diagnosed with Type 2 diabetes, fasting blood glucose 8.2 mmol/L, prescribed metformin 500mg twice daily", "entities_or_values": ["Tom", "Type 2 diabetes", "fasting blood glucose", "8.2 mmol/L", "metformin", "500mg", "twice daily"]}}
]
```"""
