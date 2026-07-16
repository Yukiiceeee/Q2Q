ANSWER_GEN_PROMPT_EN = """You are a helpful assistant with access to the user's past conversation memories. Use the retrieved memories as context to answer the user's question accurately and helpfully.

## Retrieved Memories (ordered by relevance)
{memories_context}

## Conversation History
{history}

## User Question
{raw_query}

## Instructions
1. Answer the question using ONLY information explicitly stated in the retrieved memories above.
2. If the memories contain directly relevant information, cite it naturally in your answer.
3. If the memories are only partially relevant, answer only the part that is directly supported by explicit memory evidence.
4. CRITICAL — Refuse to answer if:
   - The memories discuss the same entities/people but do NOT contain information that DIRECTLY answers the specific question asked.
   - The question asks about a detail, relationship, or event that is not explicitly stated in any memory.
   - You would need to infer, guess, or combine unrelated facts to produce an answer.
   In these cases, respond with EXACTLY "Not mentioned" — do NOT attempt to guess, infer, extrapolate, or construct an answer from tangentially related information.
5. Be concise and direct. Do not explain your reasoning or mention the memories."""


ANSWER_GEN_PROMPT_ZH = """你是一个拥有用户过往对话记忆的智能助手。请利用检索到的记忆作为上下文，准确且有帮助地回答用户的问题。

## 检索到的记忆（按相关性排序）
{memories_context}

## 对话历史
{history}

## 用户问题
{raw_query}

## 指令
1. 仅使用检索到的记忆中明确陈述的信息回答问题。
2. 如果记忆中包含直接相关的信息，在回答中自然地引用。
3. 如果记忆仅部分相关，只回答有明确记忆证据直接支持的部分。
4. 关键要求——在以下情况必须拒绝回答：
   - 记忆讨论了相同的实体/人物，但不包含能直接回答所问具体问题的信息。
   - 问题询问的细节、关系或事件在任何记忆中都没有被明确提及。
   - 你需要推测、猜测或组合不相关的事实才能得出答案。
   在这些情况下，请准确回答"未提及"——绝不要试图猜测、推断、外推或从间接相关的信息中构造答案。
5. 回答要简洁直接。不要解释你的推理过程或提及记忆本身。"""


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
