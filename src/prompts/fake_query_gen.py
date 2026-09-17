"""Fake Query Generation prompts - multi-dimension routing with style/role control."""

from src.prompts.query_format import get_fq_format_spec


# ======================== Style Instruction Templates ========================

STYLE_INSTRUCTIONS = {
    "formal": {
        "zh": "使用完整规范的书面问句。包含完整的时间、主体、事件、结果要素，语句通顺正式。",
        "en": "Use complete, formal written questions. Include full time, subject, event, and result elements with proper grammar.",
    },
    "colloquial": {
        "zh": "使用口语化、自然的提问方式。可使用语气词（呢、吧、来着）、缩写、非正式表达，但仍需包含关键信息要素。",
        "en": "Use casual, natural spoken language. May include filler words, contractions, and informal phrasing, but still include key information elements.",
    },
    "keyword": {
        "zh": "使用关键词组合形式。省略疑问词和完整语法结构，仅保留核心信息词汇的组合，但仍需覆盖时间+主体+事件+结果的关键词。",
        "en": "Use keyword combination form. Omit interrogatives and full grammar, keep only core information words, but still cover time+subject+event+result keywords.",
    },
    "elliptical": {
        "zh": "使用省略式提问。假设部分上下文已知，省略可从语境推断的信息，但保留足以定位答案的关键要素。",
        "en": "Use elliptical questions. Assume some context is known, omit information inferable from context, but retain key elements sufficient to locate the answer.",
    },
}

ROLE_INSTRUCTIONS = {
    "user": {
        "zh": "以第一人称（我/我的）视角提问，模拟用户回忆自己的经历。",
        "en": "Ask from the first-person perspective (I/my/me), simulating a user recalling their own experiences.",
    },
    "otherside": {
        "zh": "以第三方视角提问，使用具体人名或'该用户'等指代，模拟外部评估者的提问方式。",
        "en": "Ask from a third-party perspective, using specific names or references like 'the user', simulating an external evaluator's questioning style.",
    },
}


# ======================== Dimension Definitions ========================

DIMENSION_SPEC_ZH = """## 内容维度（8 种查询角度）
| 维度 | 查询重点 |
|------|----------|
| entity_fact | Who/What/How much 精确事实查询 |
| temporal | When/顺序/时间范围查询 |
| causal | Why/原因分析/归因查询 |
| opinion | 偏好/选择/态度查询 |
| state_change | 前后对比/趋势/变化分析查询 |
| summary | 总结/概览/How many 查询 |
| general | 模糊化/口语化/不完整查询 |
| natural_recall | 短句（5-15字）自然记忆回忆查询 |
"""

DIMENSION_SPEC_EN = """## Content Dimensions (8 query angles)
| Dimension | Query Focus |
|-----------|-------------|
| entity_fact | Who/What/How much precise factual queries |
| temporal | When/sequence/time range queries |
| causal | Why/root cause/attribution queries |
| opinion | Preference/choice/attitude queries |
| state_change | Before-after comparison/trend/change analysis queries |
| summary | Summary/overview/How many queries |
| general | Vague/colloquial/incomplete queries |
| natural_recall | Short (5-15 words), natural memory recall queries |
"""


# ======================== Main Prompt Templates ========================

FAKE_QUERY_GEN_PROMPT_ZH = """你是一个记忆索引助手。给定一组原子化命题（Propositions），生成用户未来可能提出的假想查询。

{query_format_spec}

{dimension_spec}

## 表述样式要求
{style_instruction}

## 角色视角要求
{role_instruction}

## 核心约束
1. 每条查询必须包含来源命题中的2个以上具体关键词（人名、数值、日期等）
2. 禁止生成可匹配多个不同事件的笼统查询
3. **必须同时包含第一人称和第三方视角的查询**：至少40%使用第三方视角（如"Melanie什么时候..."而非"我什么时候..."）
4. **强制要求**：至少 2 条查询必须是 temporal 维度
5. **强制要求**：至少 40% 的查询必须在15字以内——短句自然提问
6. **覆盖所有命题**：每个命题至少被一条查询覆盖

## 输入命题列表
{propositions_text}

## 生成指令
生成至少 {num_queries} 条查询，确保：
- 每条绑定答案来源命题 ID（answer_source），一个查询可绑定 1-5 个命题
- 标注维度（dimension 字段）
- 混合长短问句和不同视角

仅输出 JSON 数组：
```json
[
  {{"query": "查询文本", "answer_source": ["P1", "P3"], "dimension": "temporal"}},
  {{"query": "查询文本", "answer_source": ["P2"], "dimension": "entity_fact"}}
]
```"""


FAKE_QUERY_GEN_PROMPT_EN = """You are a memory indexing assistant. Given a set of atomic propositions, generate hypothetical queries that a user might ask in the future.

{query_format_spec}

{dimension_spec}

## Expression Style Requirement
{style_instruction}

## Role Perspective Requirement
{role_instruction}

## Core Constraints
1. Each query MUST contain 2+ specific keywords from the source propositions (person names, values, dates, etc.)
2. Do NOT generate generic queries that could match multiple unrelated events
3. **MUST include both first-person and third-person perspective queries**: at least 40% should use third-person perspective (e.g., "When did Melanie..." instead of "When did I...")
4. **MANDATORY**: At least 2 queries must be temporal dimension
5. **MANDATORY**: At least 40% of queries must be under 15 words — short, natural questions
6. **Full coverage**: EVERY proposition must be covered by at least one query

## Input Propositions
{propositions_text}

## Generation Instructions
Generate at least {num_queries} queries, ensuring:
- Each query binds to answer source proposition IDs (answer_source field), one query can bind 1-5 propositions
- Label each query's dimension (dimension field)
- Mix long and short questions, and different perspectives

Output ONLY a JSON array:
```json
[
  {{"query": "query text", "answer_source": ["P1", "P3"], "dimension": "temporal"}},
  {{"query": "query text", "answer_source": ["P2"], "dimension": "entity_fact"}}
]
```"""


def get_style_instruction(style: str, language: str) -> str:
    style_map = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["formal"])
    return style_map.get(language, style_map["en"])


def get_role_instruction(role: str, language: str) -> str:
    role_map = ROLE_INSTRUCTIONS.get(role, ROLE_INSTRUCTIONS["user"])
    return role_map.get(language, role_map["en"])


def build_fake_query_gen_prompt(
    propositions_text: str,
    num_queries: int = 10,
    language: str = "zh",
    style: str = "formal",
    role: str = "user",
) -> str:
    template = FAKE_QUERY_GEN_PROMPT_ZH if language == "zh" else FAKE_QUERY_GEN_PROMPT_EN
    dimension_spec = DIMENSION_SPEC_ZH if language == "zh" else DIMENSION_SPEC_EN

    return template.format(
        query_format_spec=get_fq_format_spec(language),
        dimension_spec=dimension_spec,
        style_instruction=get_style_instruction(style, language),
        role_instruction=get_role_instruction(role, language),
        propositions_text=propositions_text,
        num_queries=num_queries,
    )
