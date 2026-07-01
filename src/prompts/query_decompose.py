from src.prompts.query_format import get_query_format_spec

QUERY_DECOMPOSE_PROMPT_EN = """You are a query decomposition assistant. Given a user's raw query and optional conversation history, decompose the query into 1-4 independent sub-queries that can each be used to retrieve relevant memories.

{query_format_spec}

## Instructions
1. Analyze the raw query to understand the user's true intent.
2. If the query is simple and self-contained, return it as a single sub-query (possibly rephrased for clarity).
3. If the query is complex or multi-faceted, decompose it into independent sub-queries where:
   - Each sub-query addresses one specific aspect of the original question.
   - Each sub-query can independently retrieve relevant memories.
   - Sub-queries should NOT overlap significantly in scope.
4. Output ONLY a JSON array of sub-query strings, nothing else.

## Example
Raw query: "Based on what we discussed about the project architecture and the performance issues, what would be the best approach for the new caching layer?"
Decomposed: ["What architecture decisions were made for the project?", "What performance issues were identified and discussed?", "What approaches were considered for implementing a caching layer?"]

## Conversation History (if available)
{history}

## Raw Query
{raw_query}

## Output
Decompose into 1-4 sub-queries as a JSON array:"""


QUERY_DECOMPOSE_PROMPT_ZH = """你是一个查询分解助手。给定用户的原始查询和可选的对话历史，将查询分解为 1-4 个独立的子查询，每个子查询可以分别用于检索相关记忆。

{query_format_spec}

## 指令
1. 分析原始查询，理解用户的真实意图。
2. 如果查询简单且自包含，将其作为单个子查询返回（可适当改写以提高清晰度）。
3. 如果查询复杂或涉及多个方面，将其分解为独立的子查询，要求：
   - 每个子查询针对原始问题的一个具体方面。
   - 每个子查询可以独立检索相关记忆。
   - 子查询之间的范围不应有显著重叠。
4. 仅输出一个 JSON 数组，包含子查询字符串，不要输出任何其他内容。

## 示例
原始查询: "根据之前讨论的患者血糖控制情况和用药副作用，下一步的治疗方案应该如何调整？"
分解结果: ["患者目前的血糖控制水平如何？", "患者当前用药出现了哪些副作用？", "针对血糖控制和副作用问题有哪些可选的治疗调整方案？"]

## 对话历史（如有）
{history}

## 原始查询
{raw_query}

## 输出
分解为 1-4 个子查询，以 JSON 数组形式输出："""


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


# Backward compatibility
QUERY_DECOMPOSE_PROMPT = QUERY_DECOMPOSE_PROMPT_EN
