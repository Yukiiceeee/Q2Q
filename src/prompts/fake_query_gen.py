from src.prompts.query_format import get_query_format_spec

FAKE_QUERY_GEN_PROMPT_ZH = """你是一个记忆索引助手。给定一段会话内容，生成用户未来可能会提出的假想查询。这些查询将作为记忆检索的索引，需要高质量、多维度、细粒度覆盖。

{query_format_spec}

## 维度覆盖要求
生成的查询必须覆盖以下所有维度：
1. **实体事实**：涉及具体人物、具体事件、数值指标等客观事实的查询。
2. **时间定位**：涉及"何时"发生某事、事件的时间顺序、具体日期或阶段的查询。
3. **状态变化**：涉及变化趋势（升高/降低/稳定）、行为调整前后对比的查询。
4. **因果关系**：涉及"为什么"做出某个决策、某个结果的原因分析、归因查询。

## 指令
1. 仔细阅读会话内容，提取所有关键事实、数值、决策和变化。
2. 按照上述四个维度，为每个维度生成尽可能多的假想查询，总数至少 {num_queries} 条。如果会话信息密度高，可以生成更多。
3. 确保查询具体、可回答，且覆盖会话中所有重要信息，不要遗漏关键事实。
4. 每条查询按"时间+主体+事件+结果"结构组织，包含具体细节（特定名词、数值、时间节点等）。
5. 仅输出一个 JSON 数组，包含查询字符串，不要输出任何其他内容。

## 会话内容
{session_text}

## 输出
生成假想查询，以 JSON 数组形式输出："""


FAKE_QUERY_GEN_PROMPT_EN = """You are a memory indexing assistant. Given a conversation session, generate hypothetical queries that a user might ask in the future. These queries will serve as retrieval indices for the memory, requiring high quality, multi-dimensional, and fine-grained coverage.

{query_format_spec}

## Dimension Coverage Requirements
Generated queries MUST cover ALL of the following dimensions:
1. **Entity Facts**: Queries about specific people, specific events, numeric indicators, and other objective facts.
2. **Temporal Localization**: Queries about "when" something happened, event chronology, specific dates or phases.
3. **State Changes**: Queries about trends (increase/decrease/stable), before/after comparisons of adjustments or actions.
4. **Causal Relationships**: Queries about "why" a decision was made, root cause analysis, attribution of outcomes.

## Instructions
1. Read the session content carefully, extracting all key facts, values, decisions, and changes.
2. For each of the four dimensions above, generate as many hypothetical queries as possible, with a minimum of {num_queries} total. Generate more if the session is information-dense.
3. Each query should follow "time + subject + event + result" structure with specific details (names, values, time points).
4. Ensure queries are specific, answerable, and cover ALL important information in the session without omitting key facts.
5. Output ONLY a JSON array of query strings, nothing else.

## Session Content
{session_text}

## Output
Generate hypothetical queries, as a JSON array:"""


def build_fake_query_gen_prompt(
    session_text: str,
    num_queries: int = 10,
    language: str = "zh",
) -> str:
    template = FAKE_QUERY_GEN_PROMPT_ZH if language == "zh" else FAKE_QUERY_GEN_PROMPT_EN
    return template.format(
        query_format_spec=get_query_format_spec(language),
        session_text=session_text,
        num_queries=num_queries,
    )


FAKE_QUERY_GEN_PROMPT = FAKE_QUERY_GEN_PROMPT_EN
