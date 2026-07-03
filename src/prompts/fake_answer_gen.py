import json


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
