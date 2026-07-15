ANSWER_GEN_PROMPT_EN = """You are a helpful assistant with access to the user's past conversation memories. Use the retrieved memories as context to answer the user's question accurately and helpfully.

## Retrieved Memories (ordered by relevance)
{memories_context}

## Conversation History
{history}

## User Question
{raw_query}

## Instructions
1. Answer the question using ONLY information from the retrieved memories.
2. If the memories contain directly relevant information, cite it naturally in your answer.
3. If the memories are only partially relevant, answer only the part that is supported by memory evidence.
4. If none of the memories contain information that directly answers the question, respond with "Not mentioned" or "No information available" — do NOT guess or infer from general knowledge.
5. Be concise and direct."""


ANSWER_GEN_PROMPT_ZH = """你是一个拥有用户过往对话记忆的智能助手。请利用检索到的记忆作为上下文，准确且有帮助地回答用户的问题。

## 检索到的记忆（按相关性排序）
{memories_context}

## 对话历史
{history}

## 用户问题
{raw_query}

## 指令
1. 仅使用检索到的记忆中的信息回答问题。
2. 如果记忆中包含直接相关的信息，在回答中自然地引用。
3. 如果记忆仅部分相关，只回答有记忆证据支持的部分。
4. 如果检索到的记忆中没有能直接回答该问题的信息，明确回答"未提及"或"没有相关记录"——绝不要根据推测或常识编造答案。
5. 回答要简洁直接。"""


def build_answer_gen_prompt(
    raw_query: str,
    memories_context: str,
    history: str = "",
    language: str = "zh",
) -> str:
    template = ANSWER_GEN_PROMPT_ZH if language == "zh" else ANSWER_GEN_PROMPT_EN
    if language == "zh":
        history_text = history if history else "（无对话历史）"
    else:
        history_text = history if history else "(No conversation history)"
    return template.format(
        memories_context=memories_context,
        history=history_text,
        raw_query=raw_query,
    )


# Backward compatibility
ANSWER_GEN_PROMPT = ANSWER_GEN_PROMPT_EN
