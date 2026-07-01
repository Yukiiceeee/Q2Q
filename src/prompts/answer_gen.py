ANSWER_GEN_PROMPT_EN = """You are a helpful assistant with access to the user's past conversation memories. Use the retrieved memories as context to answer the user's question accurately and helpfully.

## Retrieved Memories (ordered by relevance)
{memories_context}

## Conversation History
{history}

## User Question
{raw_query}

## Instructions
1. Answer the question using information from the retrieved memories when relevant.
2. If the memories contain directly relevant information, cite it naturally in your answer.
3. If the memories are only partially relevant, use them as context but rely on your own knowledge to fill gaps.
4. If none of the memories are relevant, answer based on your own knowledge and note that no relevant past conversations were found.
5. Be concise and direct."""


ANSWER_GEN_PROMPT_ZH = """你是一个拥有用户过往对话记忆的智能助手。请利用检索到的记忆作为上下文，准确且有帮助地回答用户的问题。

## 检索到的记忆（按相关性排序）
{memories_context}

## 对话历史
{history}

## 用户问题
{raw_query}

## 指令
1. 利用检索到的记忆中的信息回答问题。
2. 如果记忆中包含直接相关的信息，在回答中自然地引用。
3. 如果记忆仅部分相关，将其作为背景参考，同时用自身知识补充。
4. 如果没有相关记忆，基于自身知识回答，并说明未找到相关的历史对话记录。
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
