import json
from typing import List


FAKE_ANSWER_GEN_PROMPT_ZH = """你是一个记忆索引助手。给定一段会话内容和一组针对该会话的查询，为每个查询生成一段信息密集的事实性答案序列。

## 指令
1. 仔细阅读会话内容。
2. 对于每个查询，从会话中提取能直接回答该查询的关键事实。
3. 答案要求：
   - 信息密集：用最少的字数包含最多的事实性信息（数值、时间、实体、因果关系等）。
   - 结构化：直接陈述事实，包含具体的事件、时间、人物、因果逻辑。
   - 简短精练：每个答案 20-100 字，不要冗余修饰和客套话。
   - 独立可理解：无需参考上下文即可理解答案含义。
4. 如果会话中没有某个查询的答案信息，返回空字符串 ""。
5. 输出格式为 JSON 字符串数组，每个元素对应一个查询的答案，顺序与输入查询一致。

## 会话内容
{session_text}

## 查询列表
{queries_json}

## 输出
为每个查询生成事实性答案，以 JSON 字符串数组形式输出（顺序与查询一一对应）："""


FAKE_ANSWER_GEN_PROMPT_EN = """You are a memory indexing assistant. Given a conversation session and a set of queries about it, generate dense factual answer sequences for each query.

## Instructions
1. Read the session content carefully.
2. For each query, extract the key facts from the session that directly answer it.
3. Answer requirements:
   - Information-dense: maximum facts in minimum words (values, times, names, causal links).
   - Structured: state facts directly with specific values, times, people, causal logic.
   - Concise: 20-100 words per answer, no filler or pleasantries.
   - Self-contained: understandable without additional context.
4. If the session does not contain information to answer a query, return an empty string "".
5. Output a JSON array of answer strings, in the same order as the input queries.

## Session Content
{session_text}

## Query List
{queries_json}

## Output
Generate factual answers as a JSON string array (matching query order):"""


def build_fake_answer_gen_prompt(
    session_text: str,
    queries: list[str],
    language: str = "zh",
) -> str:
    template = FAKE_ANSWER_GEN_PROMPT_ZH if language == "zh" else FAKE_ANSWER_GEN_PROMPT_EN
    queries_json = json.dumps(queries, ensure_ascii=False, indent=2)
    return template.format(
        session_text=session_text,
        queries_json=queries_json,
    )


# ============================================================================
# Version-aware answer generation prompts
# ============================================================================

FAKE_ANSWER_GEN_WITH_HISTORY_PROMPT_ZH = """你是一个记忆索引助手。给定一段新的会话内容、一组查询及其历史版本（含旧答案），为每个查询生成一段**更新后的综合答案**。

## 指令
1. 仔细阅读新会话内容。
2. 对于每个查询，参考其历史版本的旧答案，结合新会话中的最新信息，生成综合答案。
3. 综合答案要求：
   - 以新会话信息为主，历史答案为辅助参考。
   - 如果新信息与历史答案矛盾，以新信息为准，并标注"（更新）"。
   - 如果新信息是对历史的补充/延续，将两者整合为连贯的事实陈述。
   - 信息密集：包含所有相关事实（新旧合并），数值、时间、实体、因果关系。
   - 简短精练：每个答案 30-150 字。
   - 独立可理解：无需参考上下文即可理解。
4. 输出格式为 JSON 字符串数组，顺序与输入查询一致。

## 新会话内容
{session_text}

## 查询及其历史版本
{queries_with_history_json}

## 输出
为每个查询生成综合更新答案，以 JSON 字符串数组形式输出（顺序与查询一一对应）："""


FAKE_ANSWER_GEN_WITH_HISTORY_PROMPT_EN = """You are a memory indexing assistant. Given new session content, a set of queries, and their historical versions (with old answers), generate an **updated comprehensive answer** for each query.

## Instructions
1. Read the new session content carefully.
2. For each query, reference its historical answers and combine with the latest information from the new session.
3. Answer requirements:
   - Prioritize new session information; use historical answers as supplementary reference.
   - If new information contradicts history, use the new info and mark "(updated)".
   - If new information extends/continues history, integrate both into a coherent factual statement.
   - Information-dense: include all relevant facts (old + new merged), values, times, entities, causal links.
   - Concise: 30-150 words per answer.
   - Self-contained: understandable without additional context.
4. Output a JSON array of answer strings, in the same order as the input queries.

## New Session Content
{session_text}

## Queries With Historical Versions
{queries_with_history_json}

## Output
Generate comprehensive updated answers as a JSON string array (matching query order):"""


def build_fake_answer_gen_with_history_prompt(
    session_text: str,
    queries: List[str],
    version_histories: List[dict],
    language: str = "zh",
) -> str:
    """Build prompt for version-aware answer generation.

    version_histories is a parallel list to queries, each containing:
    {"related_history": [{"text": str, "answer": str, "score": float}, ...]}
    """
    template = (FAKE_ANSWER_GEN_WITH_HISTORY_PROMPT_ZH if language == "zh"
                else FAKE_ANSWER_GEN_WITH_HISTORY_PROMPT_EN)

    queries_with_history = []
    for query, vh in zip(queries, version_histories):
        entry = {
            "query": query,
            "history": [
                {
                    "old_query": h["text"],
                    "old_answer": h["answer"],
                    "similarity": round(h["score"], 3),
                }
                for h in vh.get("related_history", [])
            ],
        }
        queries_with_history.append(entry)

    queries_with_history_json = json.dumps(
        queries_with_history, ensure_ascii=False, indent=2
    )

    return template.format(
        session_text=session_text,
        queries_with_history_json=queries_with_history_json,
    )
