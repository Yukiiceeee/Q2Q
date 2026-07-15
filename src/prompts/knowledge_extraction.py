import json


KNOWLEDGE_EXTRACTION_PROMPT_ZH = """你是一个精确的医疗知识提取助手。给定一段医疗对话记录，从中提取所有离散的知识点。

## 知识点格式
每个知识点必须包含以下四个字段：
- **time**: 事件发生的时间（精确到日期，如 "2024-01-15"；若无法确定具体日期则写时间范围或阶段）
- **subject**: 知识点的主体/角色（如 "患者"、"医生"、"检查结果"、"用药方案" 等）
- **fact**: 完整的事实陈述（一句话描述发生了什么，必须具体、可验证）
- **entities_or_values**: 涉及的关键实体或数值（如药名、剂量、检查数值、症状名称等，用逗号分隔）

## 提取要求
1. **穷尽提取**：必须提取会话中每一个有意义的事实，不遗漏任何关键信息。宁可多提取，不可漏提取。
2. **精确具体**：每个 fact 必须包含具体的数值、时间、实体名称，不要模糊概括。
3. **原子化**：每个知识点只包含一个独立事实，不要将多个事实合并。
4. **全维度覆盖**：
   - 症状与体征（出现/消失/变化）
   - 检查与检验结果（具体数值）
   - 用药方案（药名、剂量、频次、变更）
   - 医嘱与建议（具体行动建议）
   - 生活方式记录（饮食、运动、睡眠、作息）
   - 时间线事件（首次出现、变化节点、就诊日期）
   - 因果关系（某行为导致某结果）
5. **数量要求**：信息密集的会话应提取 15-40 个知识点，简短会话至少提取 5-10 个。

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


KNOWLEDGE_EXTRACTION_PROMPT_EN = """You are a precise medical knowledge extraction assistant. Given a medical conversation record, extract ALL discrete knowledge points from it.

## Knowledge Point Format
Each knowledge point MUST contain these four fields:
- **time**: When the event occurred (exact date like "2024-01-15"; use time range or phase if date is uncertain)
- **subject**: The subject/role of the knowledge point (e.g., "patient", "doctor", "test result", "medication plan")
- **fact**: Complete factual statement (one sentence describing what happened, must be specific and verifiable)
- **entities_or_values**: Key entities or values involved (drug names, dosages, test values, symptom names, comma-separated)

## Extraction Requirements
1. **Exhaustive extraction**: Extract EVERY meaningful fact from the conversation. Better to over-extract than to miss anything.
2. **Precise and specific**: Each fact MUST contain specific values, times, and entity names. No vague summaries.
3. **Atomic**: Each knowledge point contains ONE independent fact only. Do NOT merge multiple facts.
4. **Full dimensional coverage**:
   - Symptoms and signs (onset/resolution/changes)
   - Test and examination results (specific values)
   - Medication plans (drug name, dosage, frequency, changes)
   - Medical advice and recommendations (specific actions)
   - Lifestyle records (diet, exercise, sleep, routine)
   - Timeline events (first occurrence, change points, visit dates)
   - Causal relationships (action X led to outcome Y)
5. **Quantity**: Information-dense sessions should yield 15-40 knowledge points; short sessions at least 5-10.

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


KNOWLEDGE_EXTRACTION_WITH_CONTEXT_PROMPT_ZH = """你是一个精确的医疗知识提取助手。给定一段新的医疗对话记录以及此前已记录的相关知识点，从新会话中提取所有新增或更新的知识点。

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


KNOWLEDGE_EXTRACTION_WITH_CONTEXT_PROMPT_EN = """You are a precise medical knowledge extraction assistant. Given a new medical conversation record and previously recorded related knowledge points, extract all NEW or UPDATED knowledge points from the new session.

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
