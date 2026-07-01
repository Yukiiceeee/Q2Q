from src.prompts.query_format import get_query_format_spec

FAKE_QUERY_GEN_PROMPT_EN = """You are a memory indexing assistant. Given a conversation session, generate hypothetical queries that a user might ask in the future, which could be directly answered using information from this session.

{query_format_spec}

## Instructions
1. Read the session content carefully.
2. Identify the key information, decisions, facts, preferences, and experiences contained in the session.
3. Generate {num_queries} hypothetical queries that:
   - Cover the most important and reusable information in the session.
   - Represent diverse angles (factual recall, how-to, preference, experience, reasoning).
   - Could realistically be asked by the same user in future conversations.
4. Output ONLY a JSON array of query strings, nothing else.

## Example Output
["What configuration was used for the database connection?", "How was the authentication flow implemented?", "Why was React chosen over Vue for the frontend?"]

## Session Content
{session_text}

## Output
Generate exactly {num_queries} hypothetical queries as a JSON array:"""


FAKE_QUERY_GEN_PROMPT_ZH = """你是一个记忆索引助手。给定一段会话内容，生成用户未来可能会提出的假想查询，这些查询可以直接使用本次会话中的信息来回答。

{query_format_spec}

## 指令
1. 仔细阅读会话内容。
2. 识别会话中包含的关键信息、决策、事实、偏好和经验。
3. 生成 {num_queries} 条假想查询，要求：
   - 覆盖会话中最重要且可复用的信息。
   - 代表不同角度（事实回忆、操作方法、偏好、经验、推理判断）。
   - 是同一用户在未来对话中可能真实提出的问题。
4. 仅输出一个 JSON 数组，包含查询字符串，不要输出任何其他内容。

## 输出示例
["患者目前的血糖控制情况如何？", "医生建议的用药调整方案是什么？", "患者的主要症状有哪些变化？"]

## 会话内容
{session_text}

## 输出
生成恰好 {num_queries} 条假想查询，以 JSON 数组形式输出："""


def build_fake_query_gen_prompt(
    session_text: str,
    num_queries: int = 5,
    language: str = "zh",
) -> str:
    template = FAKE_QUERY_GEN_PROMPT_ZH if language == "zh" else FAKE_QUERY_GEN_PROMPT_EN
    return template.format(
        query_format_spec=get_query_format_spec(language),
        session_text=session_text,
        num_queries=num_queries,
    )


# Backward compatibility
FAKE_QUERY_GEN_PROMPT = FAKE_QUERY_GEN_PROMPT_EN
