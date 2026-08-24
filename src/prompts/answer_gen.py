"""Two-stage answer generation prompts."""


ANSWER_GEN_STAGE1_PROMPT_ZH = """你是一个拥有用户过往对话记忆的智能助手。根据检索到的结构化命题，回答用户的问题。

## 检索到的记忆命题（按相关性排序）
{propositions_context}

## 对话历史
{history}

## 用户问题
{raw_query}

## 回答要求
1. **事实驱动**：仅基于上方提供的命题中的具体事实来回答，不要编造信息。
2. **推理过程**：识别与问题直接相关的命题，如果需要综合多个命题则逐步推导。
3. **回答规则**：
   - 如果命题中明确包含答案所需信息 → 直接回答，简洁准确
   - 如果命题信息不足以充分回答问题 → 在回答末尾添加标记 [INSUFFICIENT]
4. **格式**：直接给出答案，不要解释推理过程，不要提及"命题"或"记忆"。
"""


ANSWER_GEN_STAGE1_PROMPT_EN = """You are a helpful assistant with access to the user's past conversation memories. Answer the user's question based on the retrieved propositions.

## Retrieved Memory Propositions (ordered by relevance)
{propositions_context}

## Conversation History
{history}

## User Question
{raw_query}

## Answer Requirements
1. **Fact-driven**: Answer ONLY based on the propositions above. Do NOT fabricate information.
2. **Reasoning**: Identify propositions directly relevant to the question. Synthesize step by step if needed.
3. **Answer rules**:
   - If propositions clearly contain the needed information → answer directly, concisely and accurately
   - If propositions are INSUFFICIENT to fully answer → append the marker [INSUFFICIENT] at the end
4. **Format**: Give the answer directly. Do not explain reasoning or mention "propositions" or "memories".
"""


ANSWER_GEN_STAGE2_PROMPT_ZH = """你是一个拥有用户过往对话记忆的智能助手。此前尝试仅用命题回答失败，现在补充了原始会话片段，请再次回答。

## 结构化命题
{propositions_context}

## 补充会话文本
{session_context}

## 对话历史
{history}

## 用户问题
{raw_query}

## 回答要求
1. 综合命题和会话文本中的所有信息来回答。
2. 如果仍然无法回答，直接回复"未提及"。
3. 直接给出答案，不要解释推理过程。
"""


ANSWER_GEN_STAGE2_PROMPT_EN = """You are a helpful assistant with access to the user's past conversation memories. A prior attempt using only propositions was insufficient. Now supplementary session text is provided. Please answer again.

## Structured Propositions
{propositions_context}

## Supplementary Session Text
{session_context}

## Conversation History
{history}

## User Question
{raw_query}

## Answer Requirements
1. Synthesize ALL information from both propositions and session text to answer.
2. If still unable to answer, respond with "Not mentioned".
3. Give the answer directly without explaining reasoning.
"""


def build_answer_gen_prompt_stage1(
    raw_query: str,
    propositions_context: str,
    history: str = "",
    language: str = "zh",
) -> str:
    template = ANSWER_GEN_STAGE1_PROMPT_ZH if language == "zh" else ANSWER_GEN_STAGE1_PROMPT_EN
    history_text = history if history else ("（无对话历史）" if language == "zh" else "(No history)")
    return template.format(
        propositions_context=propositions_context,
        history=history_text,
        raw_query=raw_query,
    )


def build_answer_gen_prompt_stage2(
    raw_query: str,
    propositions_context: str,
    session_context: str,
    history: str = "",
    language: str = "zh",
) -> str:
    template = ANSWER_GEN_STAGE2_PROMPT_ZH if language == "zh" else ANSWER_GEN_STAGE2_PROMPT_EN
    history_text = history if history else ("（无对话历史）" if language == "zh" else "(No history)")
    return template.format(
        propositions_context=propositions_context,
        session_context=session_context,
        history=history_text,
        raw_query=raw_query,
    )
