import json


KNOWLEDGE_EXTRACTION_PROMPT_ZH = """你是一个精确的知识提取助手。给定一段对话记录，从中提取所有离散的知识点。

## 知识点格式
每个知识点必须包含以下四个字段：
- **time**: 事件发生的时间（精确到日期，如 "2024-01-15"；若无法确定具体日期则写时间范围或阶段描述）
- **subject**: 知识点的主体/角色名（如对话中的参与者名称、讨论的对象、决策方、执行方等）
- **fact**: 完整的事实陈述（一句话描述发生了什么，必须具体、可验证）
- **entities_or_values**: 涉及的关键实体或数值（如人名、地名、机构名、具体数字、产品名称、专有名词等，用逗号分隔）

## 提取要求
1. **仔细提取**：提取会话中每个有意义的事实，不遗漏任何关键信息。
2. **精确具体**：fact 必须包含具体的数值、时间、实体名称，不要模糊概括。
3. **陈述详细**：每个知识点都要详细描述独立事实，不要遗漏关键信息。
4. **全维度覆盖**：
   - 事实陈述（涉及的人物、事件、数据、结论）
   - 时间信息（具体日期、时间顺序、持续时间、截止期限）
   - 决策与行动（做了什么决定、采取了什么行动、分配了什么任务）
   - 状态与变化（之前/之后的对比、趋势、进展、结果）
   - 因果关系（某行为导致某结果、某原因引发某变化）
   - 观点与偏好（表达的意见、喜好、评价、态度、计划意向）

## 会话内容
{session_text}

## 输出
以 JSON 数组形式输出所有知识点，不输出任何其他内容：
```json
[
  {{"time": "...", "subject": "...", "fact": "...", "entities_or_values": "..."}},
  ...
]
```"""


KNOWLEDGE_EXTRACTION_PROMPT_EN = """You are a precise knowledge extraction assistant. Given a conversation record, extract ALL discrete knowledge points from it.

## Knowledge Point Format
Each knowledge point MUST contain these four fields:
- **time**: When the event occurred (exact date like "2024-01-15"; use time range or phase description if date is uncertain)
- **subject**: The subject/role of the knowledge point (e.g., participant names, discussed entity, decision-maker, executor)
- **fact**: Complete factual statement (one sentence describing what happened, must be specific and verifiable)
- **entities_or_values**: Key entities or values involved (names, places, organizations, specific numbers, product names, proper nouns, comma-separated)

## Extraction Requirements
1. **Carefully extraction**: Extract EVERY meaningful fact from the conversation.
2. **Precise and specific**: Each fact MUST contain specific values, times, and entity names. No vague summaries.
3. **Detailed statement**: each knowledge point should describe the independent facts in detail, and do not omit the key information.
4. **Full dimensional coverage**:
   - Factual statements (people, events, data, conclusions involved)
   - Temporal information (specific dates, chronological order, durations, deadlines)
   - Decisions and actions (what was decided, what action was taken, what task was assigned)
   - States and changes (before/after comparisons, trends, progress, outcomes)
   - Causal relationships (action X led to outcome Y, cause Z triggered change W)
   - Opinions and preferences (expressed views, likes, evaluations, attitudes, planned intentions)

## Session Content
{session_text}

## Output
Output all knowledge points as a JSON array, nothing else:
```json
[
  {{"time": "...", "subject": "...", "fact": "...", "entities_or_values": "..."}},
  ...
]
```"""


KNOWLEDGE_EXTRACTION_WITH_CONTEXT_PROMPT_ZH = """你是一个精确的知识提取助手。给定一段新的对话记录以及此前已记录的相关知识点，从新会话中提取所有新增或更新的知识点。

## 知识点格式
每个知识点必须包含以下四个字段：
- **time**: 事件发生的时间（精确到日期，如 "2024-01-15"；若无法确定则写时间范围）
- **subject**: 知识点的主体/角色
- **fact**: 完整的事实陈述（一句话，具体、可验证）
- **entities_or_values**: 涉及的关键实体或数值（逗号分隔）

## 已记录的相关知识点（来自关联会话）
{linked_knowledge_points_json}

## 提取要求
1. **穷尽提取新信息**：提取新会话中出现的所有新事实，不遗漏。
2. **标注更新**：如果新会话中的信息与已记录知识点矛盾或有变化，在 fact 字段开头标注"（更新）"。
3. **避免重复**：不要重复已有知识点中完全相同的信息，只提取新增和变更内容。
4. **延续整合**：如果新信息是对已有知识的补充/延续，独立记录为新知识点（可引用已有事实作为背景）。
5. **精确具体**：每个 fact 必须包含具体数值、时间、实体名称。
6. **原子化**：每个知识点只包含一个独立事实。
7. **数量要求**：信息密集的会话应提取 15-40 个知识点，简短会话至少 5-10 个。

## 新会话内容
{session_text}

## 输出
以 JSON 数组形式输出所有新增/更新的知识点：
```json
[
  {{"time": "...", "subject": "...", "fact": "...", "entities_or_values": "..."}},
  ...
]
```"""


KNOWLEDGE_EXTRACTION_WITH_CONTEXT_PROMPT_EN = """You are a precise knowledge extraction assistant. Given a new conversation record and previously recorded related knowledge points, extract all NEW or UPDATED knowledge points from the new session.

## Knowledge Point Format
Each knowledge point MUST contain these four fields:
- **time**: When the event occurred (exact date like "2024-01-15"; use range if uncertain)
- **subject**: The subject/role of the knowledge point
- **fact**: Complete factual statement (one sentence, specific, verifiable)
- **entities_or_values**: Key entities or values involved (comma-separated)

## Previously Recorded Knowledge Points (from related sessions)
{linked_knowledge_points_json}

## Extraction Requirements
1. **Exhaustive new information**: Extract ALL new facts from the new session, missing nothing.
2. **Mark updates**: If new session info contradicts or changes existing knowledge, prefix the fact with "(updated)".
3. **Avoid duplication**: Do NOT repeat information identical to existing knowledge points. Only extract new and changed content.
4. **Extend and integrate**: If new info supplements/continues existing knowledge, record as independent new knowledge points.
5. **Precise and specific**: Each fact MUST contain specific values, times, entity names.
6. **Atomic**: Each knowledge point contains ONE independent fact only.
7. **Quantity**: Dense sessions should yield 15-40 points; short sessions at least 5-10.

## New Session Content
{session_text}

## Output
Output all new/updated knowledge points as a JSON array:
```json
[
  {{"time": "...", "subject": "...", "fact": "...", "entities_or_values": "..."}},
  ...
]
```"""


def build_knowledge_extraction_prompt(
    session_text: str,
    language: str = "zh",
) -> str:
    template = KNOWLEDGE_EXTRACTION_PROMPT_ZH if language == "zh" else KNOWLEDGE_EXTRACTION_PROMPT_EN
    return template.format(session_text=session_text)


def build_knowledge_extraction_with_context_prompt(
    session_text: str,
    linked_knowledge_points: list[dict],
    language: str = "zh",
) -> str:
    template = (KNOWLEDGE_EXTRACTION_WITH_CONTEXT_PROMPT_ZH if language == "zh"
                else KNOWLEDGE_EXTRACTION_WITH_CONTEXT_PROMPT_EN)
    linked_kps_json = json.dumps(linked_knowledge_points, ensure_ascii=False, indent=2)
    return template.format(
        session_text=session_text,
        linked_knowledge_points_json=linked_kps_json,
    )
