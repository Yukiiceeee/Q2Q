ANSWER_GEN_PROMPT_ZH = """你是一个拥有用户过往对话记忆的智能助手。根据检索到的结构化知识点，回答用户的问题。

## 检索到的记忆（按相关性排序）
{memories_context}

## 对话历史
{history}

## 用户问题
{raw_query}

## 推理与回答要求
1. **事实驱动**：仅基于上方提供的知识点中的具体事实来回答，不要编造信息。
2. **推理过程**：
   - 首先识别与问题直接相关的知识点
   - 如果需要综合多个知识点才能回答，逐步推导
   - 如果知识点中包含时间信息，注意时间顺序和上下文
   - 仔细深入地推理问题答案，不要被一些相似但错误的知识点信息干扰
3. **回答规则**：
   - 如果知识点中明确包含答案所需的信息 → 直接回答，简洁准确
   - 如果知识点中没有能够充分解答问题的任何信息 → 回答"未提及"
4. **格式**：直接给出答案，不要解释你的推理过程，不要提及"知识点"或"记忆"。
5. **注意**：知识点和历史依据很长，信息太多容易迷失，请仔细再三检查过往信息，精确找出匹配问题的内容。
"""


ANSWER_GEN_PROMPT_EN = """You are a helpful assistant with access to the user's past conversation memories. Answer the user's question based on the retrieved structured knowledge points.

## Retrieved Memories (ordered by relevance)
{memories_context}

## Conversation History
{history}

## User Question
{raw_query}

## Reasoning and Answer Requirements
1. **Fact-driven**: Answer ONLY based on specific facts from the knowledge points above. Do not fabricate information.
2. **Reasoning process**:
   - First identify knowledge points directly relevant to the question
   - If multiple knowledge points need to be synthesized, reason step by step
   - If knowledge points contain temporal information, respect chronological order and context
   - Reason the answer carefully and thoroughly, and don't be disturbed by some similar but wrong information
3. **Answer rules**:
   - If knowledge points clearly contain the needed information → answer directly, concisely and accurately
   - If knowledge points contain NO information to answer the question → respond "Not mentioned"
4. **Format**: Give the answer directly. Do not explain your reasoning process or mention "knowledge points" or "memories".
5. **Note**: the knowledge points and historical basis are very long, and too much information is easy to get lost. Please carefully and repeatedly check the past information to accurately find out the content of the matching problem.
"""


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
