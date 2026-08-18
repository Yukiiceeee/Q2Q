"""Prompt templates for generating indirect-phrasing queries.

The goal is NOT simple paraphrasing. Instead, generate queries that:
- Ask about the SAME underlying memory/fact
- Use COMPLETELY DIFFERENT semantic content and surface words
- Approach the topic from an indirect angle (e.g., asking a related
  question whose answer depends on knowing the original fact)

Example:
  Memory: "Carol has a cat named Wangcai"
  Original query: "What's Carol's cat's name?"
  Indirect query: "Would a lily plant be safe to grow in Carol's home?"
  (Answering this requires knowing Carol has a cat, since lilies are toxic to cats)
"""

INDIRECT_QUERY_PROMPT_EN = """You are a creative query reformulator for a memory retrieval experiment.

## Task
Given an original question and its answer (both derived from a conversation memory), generate exactly 5 INDIRECT queries that:
1. Would require access to the SAME memory/fact to answer
2. Use COMPLETELY DIFFERENT wording, vocabulary, and sentence structure from the original
3. Approach the topic from an oblique/tangential angle
4. Do NOT directly mention the key terms from the original question or answer

## Indirection Strategies (use one per query, in order):
1. **Implication-based**: Ask something whose answer IMPLIES or REQUIRES the original fact
   - Memory: "John takes metformin" → "Should John avoid alcohol with his daily medication?"
2. **Scenario-based**: Pose a hypothetical scenario that depends on the original fact
   - Memory: "Alice is allergic to peanuts" → "If we're choosing a restaurant for Alice, should we avoid Thai cuisine?"
3. **Consequence-based**: Ask about a downstream consequence of the fact
   - Memory: "The meeting was moved to Friday" → "Do we have a scheduling conflict with the team lunch this week?"
4. **Peripheral-detail**: Ask about a related but tangential aspect
   - Memory: "Bob drives a Tesla Model 3" → "Does Bob need to find charging stations on road trips?"
5. **Negation/contrast**: Ask what would be different if the fact were otherwise
   - Memory: "Sarah graduated from MIT" → "Would Sarah's network include many Boston-area tech professionals?"

## CRITICAL RULES
- The indirect queries must NOT contain the same key nouns/entities from the original answer
- Each query must be answerable ONLY if the memory is accessible
- Queries should sound natural, as if asked by someone who knows the context but phrases things differently
- The connection between the indirect query and the original memory must be logically sound

## Input
Original question: {original_query}
Answer: {answer}
Memory context (session excerpt): {context}

## Output
Output ONLY a JSON array of exactly 5 indirect query strings (one per strategy, in order 1-5):"""


INDIRECT_QUERY_PROMPT_ZH = """你是一个创意查询改写器，用于记忆检索实验。

## 任务
给定一个原始问题及其答案（均来自会话记忆），生成恰好 5 条间接查询，要求：
1. 回答这些查询需要访问相同的记忆/事实
2. 使用与原始问题完全不同的措辞、词汇和句式
3. 从侧面/切线角度切入话题
4. 不直接提及原始问题或答案中的关键词

## 间接策略（每条用一种，按顺序）：
1. **蕴含式**：提问一个答案蕴含或依赖原始事实的问题
   - 记忆："张三在吃二甲双胍" → "张三每天吃药期间能喝酒吗？"
2. **情境式**：提出一个依赖原始事实的假设性情境
   - 记忆："李四对花生过敏" → "给李四选餐厅的话，泰国菜是不是要避开？"
3. **后果式**：询问该事实的下游后果
   - 记忆："会议改到周五了" → "这周团队午餐会不会和其他安排冲突？"
4. **周边细节式**：询问相关但切面的方面
   - 记忆："王五开特斯拉Model 3" → "王五自驾游需要提前规划充电桩吗？"
5. **否定/对比式**：询问如果事实不同会怎样
   - 记忆："小红毕业于清华" → "小红的人脉圈里北京科技圈的人应该不少吧？"

## 关键规则
- 间接查询中不能包含原始答案中的相同关键名词/实体
- 每条查询必须只有在能访问该记忆时才能回答
- 查询应自然流畅，像是了解上下文但换了种方式发问
- 间接查询与原始记忆之间的逻辑关联必须成立

## 输入
原始问题：{original_query}
答案：{answer}
记忆上下文（会话摘要）：{context}

## 输出
仅输出一个包含恰好 5 条间接查询字符串的 JSON 数组（按策略 1-5 顺序）："""


INDIRECT_QUERY_STYLES = [
    "implication",
    "scenario",
    "consequence",
    "peripheral",
    "negation_contrast",
]


def build_indirect_query_prompt(
    original_query: str,
    answer: str,
    context: str,
    language: str = "en",
) -> str:
    template = (
        INDIRECT_QUERY_PROMPT_ZH if language == "zh"
        else INDIRECT_QUERY_PROMPT_EN
    )
    return template.format(
        original_query=original_query,
        answer=answer,
        context=context[:1500],
    )
