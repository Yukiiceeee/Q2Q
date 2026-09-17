"""Single-pass answer generation prompt."""


ANSWER_GEN_PROMPT_ZH = """你是一个拥有用户过往对话记忆的智能助手。根据检索到的记忆信息，回答用户的问题。

## 检索到的记忆信息（按相关性排序）
{context}

## 对话历史
{history}

## 用户问题
{raw_query}

## 回答要求
1. **事实驱动**：仅基于上方提供的记忆信息中的具体事实来回答，不要编造信息。
2. **推理过程**：识别与问题直接相关的信息，如果需要综合多个命题则逐步推导。
3. **严格判定规则**：
   - 如果记忆信息中明确包含答案所需信息 → 直接回答，简洁准确
   - 如果记忆信息中完全没有涉及问题所问的内容，或者信息不足以回答问题 → 严格回复"信息不充分"（不要回复任何其他内容）
4. **格式**：直接给出答案，不要解释推理过程，不要提及"命题"、"记忆"或"检索"。
5. **重要**：不要猜测或推断记忆中未明确记载的信息。如果问题问的是记忆中从未出现的事件、人物或细节，必须回复"信息不充分"。
"""


ANSWER_GEN_PROMPT_EN = """You are a helpful assistant with access to the user's past conversation memories. Answer the user's question based on the retrieved memory information.

## Retrieved Memory Information (ordered by relevance)
{context}

## Conversation History
{history}

## User Question
{raw_query}

## Answer Requirements
1. **Fact-driven**: Answer ONLY based on the memory information provided above. Do NOT fabricate information.
2. **Reasoning**: Identify information directly relevant to the question. Synthesize step by step if needed.
3. **Strict Judgment Rules**:
   - If the memory information clearly contains the needed answer → answer directly, concisely and accurately
   - If the memory information does NOT contain anything related to what is being asked, or is insufficient to answer → strictly respond with "Insufficient information" (do not respond with anything else)
4. **Format**: Give the answer directly. Do not explain reasoning or mention "propositions", "memories", or "retrieval".
5. **Important**: Do NOT guess or infer information not explicitly recorded in the memories. If the question asks about events, people, or details that never appeared in the memories, you MUST respond with "Insufficient information".
"""


def build_answer_gen_prompt(
    raw_query: str,
    context: str,
    history: str = "",
    language: str = "zh",
) -> str:
    template = ANSWER_GEN_PROMPT_ZH if language == "zh" else ANSWER_GEN_PROMPT_EN
    history_text = history if history else ("（无对话历史）" if language == "zh" else "(No history)")
    return template.format(
        context=context,
        history=history_text,
        raw_query=raw_query,
    )
