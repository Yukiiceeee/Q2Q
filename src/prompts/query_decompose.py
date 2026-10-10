from src.prompts.query_format import get_query_format_spec

QUERY_DECOMPOSE_PROMPT_ZH = """你是一个查询分解与关键词提取助手。给定用户的原始查询和可选的对话历史，将查询分解为 1-4 个独立的结构化子查询，并为每个子查询提取用于检索的关键词。

{query_format_spec}

## 指令
1. 先分析原始查询，理解用户的真实意图，思考回答原始问题需要知道什么信息（在 <think> 标签内简述）。
2. 子查询的条数取决于**回答需要几条互相独立的、被记录下来的证据**，而不取决于原始查询的字面长短。问题很短不等于只需一条证据："四年前她从哪里搬来的？"依赖三件事：搬家这件事本身、搬家的起点、以及让"四年前"可被定位的时间锚点。
3. 每条独立证据出一条子查询（通常 2-3 条；只有当单独一条记录就能完整回答时才给 1 条），要求：
   - 每个子查询必须按"时间+主体+事件+结果"结构组织，包含尽可能多的具体细节。
   - 每个子查询可以独立检索相关记忆。
   - 子查询之间的范围不应有显著重叠。
   - 简短直接，8-30 字
   - 组合起来覆盖回答原始问题所需的全部信息
4. **忠实性约束（最重要）**：子查询只能使用原始查询中出现的实体、事件和概念。绝对禁止添加原查询中没有的假设、限定词或推测（如把"研究了什么"擅自改成"最近的学术研究"）。不确定的信息宁可保持宽泛，也不要凭空缩小范围。该约束管的是**可以断言什么**，不是**可以拆到多细**：把问题所依赖的中间事实单独列出来（时间锚点、先前状态、涉及的人物）属于分解，不属于推测；而擅自加上"最近的""学术的"这类缩小范围的限定词才是。
5. **标识符逐字保留**：原始查询中出现的任何显式标识符——步骤序号（如 "Step 12"）、方括号引用 ID（如 "[29575]"）、编号、日期、数值、专有名词——必须原样出现在至少一条子查询中，不得改写、翻译或省略。若查询以步骤序号而非日期定位，则用步骤锚点替代"时间"要素。
6. 为每个子查询提取 3-6 个关键词/要点，这些关键词应是回答该查询所需的核心概念、实体名称、数值指标、时间节点等。关键词同样只能来自原始查询。
7. 输出格式：先 <think>（**最多一句话**，不得展开推理），再 JSON 数组，不要输出任何其他内容。

## 输出格式
<think>
需要知道：...（一句话）
</think>
```json
[
  {{"query": "子查询", "keywords": "关键词1, 关键词2, 关键词3"}}
]
```

## 对话历史（如有）
{history}

## 原始查询
{raw_query}

## 输出"""


QUERY_DECOMPOSE_PROMPT_EN = """You are a query decomposition and keyword extraction assistant. Given a user's raw query and optional conversation history, decompose the query into 1-4 independent structured sub-queries and extract retrieval keywords for each.

{query_format_spec}

## Instructions
1. First analyze the raw query, understand the user's true intent, and reason about what information is needed to answer the original question (inside <think> tags).
2. The number of sub-queries follows from how many **independent pieces of recorded evidence** the answer rests on, not from how long the raw query looks. A short question often still rests on several: "Where did she move from 4 years ago?" depends on the move itself, the place it started from, and the time anchor that makes "4 years ago" resolvable.
3. Emit one sub-query per such piece of evidence (2-3 is typical; use a single one only when one recorded fact answers the question completely), where:
   - Each sub-query MUST be organized following the "Time + Subject + Event + Result" structure, including as many specific details as possible.
   - Each sub-query can independently retrieve relevant memories.
   - Sub-queries should NOT have significant overlap in scope.
   - Keep them short and direct, 8-30 words.
   - Together they cover ALL information needed to answer the original question.
4. **Faithfulness constraint (MOST IMPORTANT)**: Sub-queries may ONLY use entities, events, and concepts that appear in the raw query. NEVER add assumptions, qualifiers, or speculation absent from the query (e.g., do NOT rewrite "what did X research" into "X's most recent academic research"). When uncertain, stay broad — never narrow the scope with invented details. This governs **what you may assert**, not **how finely you may split**: naming an intermediate fact the question depends on (a date anchor, a prior state, a participant) is decomposition, not speculation; inventing a narrowing qualifier like "most recent" or "academic" is.
5. **Preserve identifiers verbatim**: any explicit identifier in the raw query — step indices (e.g. "Step 12"), bracketed reference ids (e.g. "[29575]"), codes, dates, numeric values, proper nouns — MUST appear unchanged in at least one sub-query. Never paraphrase, translate, or drop them. When the query is anchored by step indices rather than dates, use the step anchor in place of the Time element.
6. Extract 3-6 keywords/key points per sub-query. These keywords should be core concepts, entity names, numeric indicators, time points, etc. needed to answer the query. Keywords must also come only from the raw query.
7. Output format: <think> first (**one sentence maximum**, do not expand the reasoning), then JSON array. Do not output anything else.

## Output Format
<think>
Need to know: ... (one sentence)
</think>
```json
[
  {{"query": "sub-query", "keywords": "keyword1, keyword2, keyword3"}}
]
```

## Conversation History (if available)
{history}

## Raw Query
{raw_query}

## Output"""


def build_query_decompose_prompt(
    raw_query: str,
    history: str = "",
    language: str = "zh",
) -> str:
    template = QUERY_DECOMPOSE_PROMPT_ZH if language == "zh" else QUERY_DECOMPOSE_PROMPT_EN
    history_text = history if history else ("（无对话历史）" if language == "zh" else "(No history)")
    return template.format(
        query_format_spec=get_query_format_spec(language),
        history=history_text,
        raw_query=raw_query,
    )


QUERY_DECOMPOSE_PROMPT = QUERY_DECOMPOSE_PROMPT_EN
