"""Fake Query Generation prompts - CoT multi-dimension routing with style/role control."""

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

DIMENSION_SPEC_ZH = """## 内容维度（7 种查询角度）
分析命题特征后，按以下维度侧重生成查询：

| 维度 | 触发条件 | 查询重点 |
|------|----------|----------|
| entity_fact | 命题含具体人物、机构、数值指标 | Who/What/How much 精确事实查询 |
| temporal | 命题含时间戳、日期、时序关系 | When/顺序/时间范围查询 |
| causal | 命题含因果词（因为/所以/导致/由于） | Why/原因分析/归因查询 |
| opinion | 命题含观点、偏好、决策、选择 | 偏好/选择/态度查询 |
| state_change | 命题含变化、趋势、对比词（升高/降低/改为） | 前后对比/趋势/变化分析查询 |
| summary | 命题数量多(>8)或涵盖范围广 | 总结/概览/How many 查询 |
| general | 通用日常信息 | 模糊化/口语化/不完整查询 |
"""

DIMENSION_SPEC_EN = """## Content Dimensions (7 query angles)
After analyzing proposition features, generate queries with emphasis on these dimensions:

| Dimension | Trigger Condition | Query Focus |
|-----------|-------------------|-------------|
| entity_fact | Propositions contain specific people, organizations, numeric indicators | Who/What/How much precise factual queries |
| temporal | Propositions contain timestamps, dates, temporal relations | When/sequence/time range queries |
| causal | Propositions contain causal words (because/therefore/caused by/due to) | Why/root cause/attribution queries |
| opinion | Propositions contain opinions, preferences, decisions, choices | Preference/choice/attitude queries |
| state_change | Propositions contain change/trend/comparison words (increased/decreased/changed to) | Before-after comparison/trend/change analysis queries |
| summary | Many propositions (>8) or broad coverage | Summary/overview/How many queries |
| general | General everyday information | Vague/colloquial/incomplete queries |
"""


# ======================== Main Prompt Templates ========================

FAKE_QUERY_GEN_PROMPT_ZH = """你是一个记忆索引助手。给定一组原子化命题（Propositions），你需要分析其内容特征，然后针对性地生成用户未来可能会提出的假想查询。

**关键原则：生成的每条查询必须足够详细和具体，包含命题中的关键细节（具体名称、数值、时间、地点等），使得该查询能通过嵌入相似度精准匹配到对应的源命题。避免生成笼统、模糊的问题。**

{query_format_spec}

{dimension_spec}

## 表述样式要求
{style_instruction}

## 角色视角要求
{role_instruction}

## 核心约束
1. 无论使用什么表述样式，每条查询都必须遵循"时间+主体+事件+结果"的信息结构。样式只影响表达方式，不影响信息完整性。
2. 每条查询必须包含来源命题中的具体实体/数值/时间等关键词，确保嵌入空间中的高相似度。
3. 禁止生成可匹配多个不同事件的笼统查询。

## 输入命题列表
{propositions_text}

## 生成指令
请按以下两步推理生成查询：

**第一步：分析命题特征，确定各维度权重**
分析上述命题的内容特征（实体密度、时间信息、因果关系、观点/决策、状态变化等），判断应侧重生成哪些维度的查询。将分析结果放在输出 JSON 的 `_reasoning` 字段中。

**第二步：按权重侧重和样式要求生成查询**
根据第一步的权重判断，侧重生成高权重维度的查询，同时确保：
- 生成至少 {num_queries} 条查询
- 每条查询绑定答案来源的命题 ID（answer_source），一个查询可绑定 1-5 个命题
- 所有命题都被至少一条查询覆盖
- 标注每条查询所属的维度（dimension 字段）
- 每条查询必须含有来源命题中的2个以上具体关键词（人名、数值、日期、药物名等）

## 输出格式
仅输出 JSON，格式如下：
```json
{{
  "_reasoning": "命题特征分析：...。权重判断：entity_fact=高, temporal=高, causal=低, ...",
  "queries": [
    {{"query": "查询文本", "answer_source": ["P1", "P3"], "dimension": "temporal"}},
    {{"query": "查询文本", "answer_source": ["P2"], "dimension": "entity_fact"}}
  ]
}}
```"""


FAKE_QUERY_GEN_PROMPT_EN = """You are a memory indexing assistant. Given a set of atomic propositions, you need to analyze their content features, then generate targeted hypothetical queries that a user might ask in the future.

**KEY PRINCIPLE: Each generated query must be sufficiently detailed and specific, containing key details from the propositions (specific names, values, dates, locations, etc.), so that the query can precisely match the corresponding source propositions via embedding similarity. Avoid generating vague or generic questions.**

{query_format_spec}

{dimension_spec}

## Expression Style Requirement
{style_instruction}

## Role Perspective Requirement
{role_instruction}

## Core Constraints
1. Regardless of expression style, every query MUST follow the "Time + Subject + Event + Result" information structure. Style only affects HOW queries are expressed, NOT information completeness.
2. Each query MUST contain specific entities/values/timestamps/keywords from the source propositions to ensure high similarity in embedding space.
3. Do NOT generate generic queries that could match multiple unrelated events.

## Input Propositions
{propositions_text}

## Generation Instructions
Generate queries using the following two-step reasoning:

**Step 1: Analyze proposition features and determine dimension weights**
Analyze the content features of the propositions above (entity density, temporal information, causal relationships, opinions/decisions, state changes, etc.) and determine which dimensions to emphasize. Put the analysis in the `_reasoning` field of the output JSON.

**Step 2: Generate queries according to weights and style requirements**
Based on Step 1's weight judgment, emphasize high-weight dimensions while ensuring:
- Generate at least {num_queries} queries
- Each query binds to answer source proposition IDs (answer_source field), one query can bind 1-5 propositions
- ALL propositions are covered by at least one query
- Label each query's dimension (dimension field)
- Each query MUST contain 2+ specific keywords from the source propositions (person names, values, dates, drug names, etc.)

## Output Format
Output ONLY JSON in this format:
```json
{{
  "_reasoning": "Proposition feature analysis: ... Weight judgment: entity_fact=high, temporal=high, causal=low, ...",
  "queries": [
    {{"query": "query text", "answer_source": ["P1", "P3"], "dimension": "temporal"}},
    {{"query": "query text", "answer_source": ["P2"], "dimension": "entity_fact"}}
  ]
}}
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
