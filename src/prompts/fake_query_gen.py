"""Fake Query Generation prompts - proposition-based."""

from src.prompts.query_format import get_query_format_spec


FAKE_QUERY_GEN_PROMPT_ZH = """你是一个记忆索引助手。给定一组原子化命题（Propositions），生成用户未来可能会提出的假想查询。每个查询必须明确绑定其答案来源的命题 ID。

{query_format_spec}

## 维度覆盖要求
生成的查询必须覆盖以下维度：
1. **实体事实**：涉及具体人物、事件、数值指标的查询
2. **时间定位**：涉及"何时"、时间顺序、具体日期的查询
3. **状态变化**：涉及变化趋势、前后对比的查询
4. **因果关系**：涉及"为什么"、原因分析的查询

## 输入命题列表
{propositions_text}

## 指令
1. 分析所有命题，理解其中包含的事实信息。
2. 为这些命题生成至少 {num_queries} 条假想查询，覆盖上述四个维度。
3. 每条查询必须绑定其答案来源的命题 ID（answer_source 字段），一个查询可以绑定 1-5 个命题。
4. 确保查询具体、可回答，且覆盖所有重要信息。
5. 仅输出 JSON 数组。

## 输出格式
```json
[
  {{"query": "查询文本", "answer_source": ["P1", "P3"]}},
  {{"query": "查询文本", "answer_source": ["P2"]}}
]
```"""


FAKE_QUERY_GEN_PROMPT_EN = """You are a memory indexing assistant. Given a set of atomic propositions, generate hypothetical queries that a user might ask in the future. Each query MUST explicitly bind to its answer source proposition IDs.

{query_format_spec}

## Dimension Coverage Requirements
Generated queries MUST cover:
1. **Entity Facts**: Queries about specific people, events, numeric indicators
2. **Temporal Localization**: Queries about "when", chronological order, specific dates
3. **State Changes**: Queries about trends, before/after comparisons
4. **Causal Relationships**: Queries about "why", root cause analysis

## Input Propositions
{propositions_text}

## Instructions
1. Analyze all propositions to understand the factual information.
2. Generate at least {num_queries} hypothetical queries covering the four dimensions above.
3. Each query MUST bind to its answer source proposition IDs (answer_source field). One query can bind to 1-5 propositions.
4. Ensure queries are specific, answerable, and cover all important information.
5. Output ONLY a JSON array.

## Output Format
```json
[
  {{"query": "query text", "answer_source": ["P1", "P3"]}},
  {{"query": "query text", "answer_source": ["P2"]}}
]
```"""


def build_fake_query_gen_prompt(
    propositions_text: str,
    num_queries: int = 10,
    language: str = "zh",
) -> str:
    template = FAKE_QUERY_GEN_PROMPT_ZH if language == "zh" else FAKE_QUERY_GEN_PROMPT_EN
    return template.format(
        query_format_spec=get_query_format_spec(language),
        propositions_text=propositions_text,
        num_queries=num_queries,
    )
