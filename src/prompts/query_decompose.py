from src.prompts.query_format import get_query_format_spec

QUERY_DECOMPOSE_PROMPT_ZH = """你是一个查询分解与关键词提取助手。给定用户的原始查询和可选的对话历史，将查询分解为 1-4 个独立的结构化子查询，并为每个子查询提取用于检索的关键词。

{query_format_spec}

## 指令
1. 先分析原始查询，理解用户的真实意图，思考回答原始问题需要知道什么信息（在 <think> 标签内简述）。
2. 如果查询简单且自包含，将其作为单个子查询返回，改写为按"时间+主体+事件+结果"结构组织。
3. 如果查询复杂或涉及多个方面，将其分解为独立的子查询，要求：
   - 每个子查询必须按"时间+主体+事件+结果"结构组织，包含尽可能多的具体细节。
   - 每个子查询可以独立检索相关记忆。
   - 子查询之间的范围不应有显著重叠。
   - 简短直接，8-30 字
   - 组合起来覆盖回答原始问题所需的全部信息
4. 为每个子查询提取 3-6 个关键词/要点，这些关键词应是回答该查询所需的核心概念、实体名称、数值指标、时间节点等。
5. 输出格式：先 <think>，再 JSON 数组，不要输出任何其他内容。

## 输出格式
<think>
回答此问题需要知道：...
</think>
```json
[
  {{"query": "子查询", "keywords": "关键词1, 关键词2, 关键词3"}}
]
```

## 对话历史（如有）
{history}

## 原始查询
{raw_query}

## 输出"""


QUERY_DECOMPOSE_PROMPT_EN = """You are a query decomposition and keyword extraction assistant. Given a user's raw query and optional conversation history, decompose the query into 1-4 independent structured sub-queries and extract retrieval keywords for each.

{query_format_spec}

## Instructions
1. First analyze the raw query, understand the user's true intent, and reason about what information is needed to answer the original question (inside <think> tags).
2. If the query is simple and self-contained, return it as a single sub-query, rewritten to follow the "Time + Subject + Event + Result" structure.
3. If the query is complex or involves multiple aspects, decompose it into independent sub-queries where:
   - Each sub-query MUST be organized following the "Time + Subject + Event + Result" structure, including as many specific details as possible.
   - Each sub-query can independently retrieve relevant memories.
   - Sub-queries should NOT have significant overlap in scope.
   - Keep them short and direct, 8-30 words.
   - Together they cover ALL information needed to answer the original question.
4. Extract 3-6 keywords/key points per sub-query. These keywords should be core concepts, entity names, numeric indicators, time points, etc. needed to answer the query.
5. Output format: <think> first, then JSON array. Do not output anything else.

## Output Format
<think>
To answer this question I need to know: ...
</think>
```json
[
  {{"query": "sub-query", "keywords": "keyword1, keyword2, keyword3"}}
]
```

## Conversation History (if available)
{history}

## Raw Query
{raw_query}

## Output"""


def build_query_decompose_prompt(
    raw_query: str,
    history: str = "",
    language: str = "zh",
) -> str:
    template = QUERY_DECOMPOSE_PROMPT_ZH if language == "zh" else QUERY_DECOMPOSE_PROMPT_EN
    history_text = history if history else ("（无对话历史）" if language == "zh" else "(No history)")
    return template.format(
        query_format_spec=get_query_format_spec(language),
        history=history_text,
        raw_query=raw_query,
    )


QUERY_DECOMPOSE_PROMPT = QUERY_DECOMPOSE_PROMPT_EN
