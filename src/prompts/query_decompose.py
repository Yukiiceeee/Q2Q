from src.prompts.query_format import get_query_format_spec

QUERY_DECOMPOSE_PROMPT_ZH = """你是一个查询分解与关键词提取助手。给定用户的原始查询和可选的对话历史，将查询分解为 1-4 个独立的结构化子查询，并为每个子查询提取用于检索的关键词。

{query_format_spec}

## 指令
1. 分析原始查询，理解用户的真实意图。
2. 如果查询简单且自包含，将其作为单个子查询返回（按结构化格式改写）。
3. 如果查询复杂或涉及多个方面，将其分解为独立的子查询，要求：
   - 每个子查询按"时间+主体+事件+结果"结构组织，包含尽可能多的具体细节。
   - 每个子查询可以独立检索相关记忆。
   - 子查询之间的范围不应有显著重叠。
4. 为每个子查询提取 3-6 个关键词/要点，这些关键词应是回答该查询所需的核心概念、实体名称、数值指标、时间节点等。
5. 仅输出一个 JSON 数组，每个元素为对象 {{"query": "...", "keywords": "..."}}，不要输出任何其他内容。

## 输出格式
```json
[
  {{"query": "结构化子查询1", "keywords": "关键词1, 关键词2, 关键词3"}},
  {{"query": "结构化子查询2", "keywords": "关键词1, 关键词2, 关键词3"}}
]
```

## 对话历史（如有）
{history}

## 原始查询
{raw_query}

## 输出
分解为 1-4 个子查询，以 JSON 数组形式输出："""


QUERY_DECOMPOSE_PROMPT_EN = """You are a query decomposition and keyword extraction assistant. Given a user's raw query and optional conversation history, decompose the query into 1-4 independent structured sub-queries, and extract retrieval keywords for each.

{query_format_spec}

## Instructions
1. Analyze the raw query to understand the user's true intent.
2. If the query is simple and self-contained, return it as a single sub-query (rewritten in structured format).
3. If the query is complex or multi-faceted, decompose it into independent sub-queries where:
   - Each sub-query is organized as "time + subject + event + result", with as many specific details as possible.
   - Each sub-query can independently retrieve relevant memories.
   - Sub-queries should NOT overlap significantly in scope.
4. For each sub-query, extract 3-6 keywords/key-points: core concepts, entity names, numeric indicators, time points needed to answer it.
5. Output ONLY a JSON array where each element is {{"query": "...", "keywords": "..."}}, nothing else.

## Output Format
```json
[
  {{"query": "structured sub-query 1", "keywords": "keyword1, keyword2, keyword3"}},
  {{"query": "structured sub-query 2", "keywords": "keyword1, keyword2, keyword3"}}
]
```

## Conversation History (if available)
{history}

## Raw Query
{raw_query}

## Output
Decompose into 1-4 sub-queries as a JSON array:"""


def build_query_decompose_prompt(
    raw_query: str,
    history: str = "",
    language: str = "zh",
) -> str:
    template = QUERY_DECOMPOSE_PROMPT_ZH if language == "zh" else QUERY_DECOMPOSE_PROMPT_EN
    history_text = history if history else ("（无对话历史）" if language == "zh" else "(No conversation history available)")
    return template.format(
        query_format_spec=get_query_format_spec(language),
        history=history_text,
        raw_query=raw_query,
    )


QUERY_DECOMPOSE_PROMPT = QUERY_DECOMPOSE_PROMPT_EN
