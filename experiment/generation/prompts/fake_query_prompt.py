"""Prompt templates for generating fake queries in the experiment context.

These prompts are independent of the main Q2Q system prompts -- they are
designed specifically for the motivation analysis experiment where we need
FQ generation that is completely blind to the true_query.
"""

FAKE_QUERY_GEN_EXPERIMENT_EN = """You are a memory indexing assistant. Given a conversation session, generate hypothetical queries that a user might ask in the future about the information discussed in this session.

## Requirements
1. Generate exactly {num_queries} diverse hypothetical queries.
2. Queries must cover ALL important facts, events, decisions, and details mentioned in the session.
3. Each query should follow a structured question form: time + subject + event/action + result/state.
4. Use interrogative words (What/How/Why/When/Where/Which/Can/Does/Is).
5. Be specific — include names, values, dates, and concrete details from the session.
6. Do NOT reference "the conversation" or "the session" — treat each query as a standalone question.

## Dimension Coverage
Ensure queries span these dimensions:
- Entity Facts: specific people, events, numeric values
- Temporal: when things happened, chronological order
- State Changes: trends, before/after comparisons
- Causal: why decisions were made, root causes

## Session Content
{session_text}

## Output
Output ONLY a JSON array of {num_queries} query strings:"""


FAKE_QUERY_GEN_EXPERIMENT_ZH = """你是一个记忆索引助手。给定一段会话内容，生成用户未来可能会提出的假想查询。

## 要求
1. 生成恰好 {num_queries} 条多样化的假想查询。
2. 查询必须覆盖会话中提到的所有重要事实、事件、决策和细节。
3. 每条查询按"时间+主体+事件+结果"结构组织。
4. 以疑问词开头（什么/如何/为什么/何时/哪里/哪个/是否）。
5. 具体明确——包含会话中的人名、数值、日期等具体细节。
6. 不要提及"对话"或"会话"——视为独立问题。

## 维度覆盖
确保查询涵盖：
- 实体事实：具体人物、事件、数值指标
- 时间定位：何时发生、时间顺序
- 状态变化：趋势、前后对比
- 因果关系：为什么做出决策、归因

## 会话内容
{session_text}

## 输出
仅输出一个包含 {num_queries} 条查询字符串的 JSON 数组："""


def build_experiment_fq_prompt(
    session_text: str,
    num_queries: int = 10,
    language: str = "en",
) -> str:
    template = (
        FAKE_QUERY_GEN_EXPERIMENT_ZH if language == "zh"
        else FAKE_QUERY_GEN_EXPERIMENT_EN
    )
    return template.format(session_text=session_text, num_queries=num_queries)
