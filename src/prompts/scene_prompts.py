"""Scene-specific prompt variants for proposition extraction and FQ generation.

The dialogue scene keeps the original prompts in `proposition_extractor` /
`fake_query_gen`. This module supplies the two additional scenes — agent
execution trajectories and generic documents — so that mode switching never
touches the dialogue path that is already tuned.
"""
from __future__ import annotations

from src.scenes import SCENE_DIALOGUE, SCENE_DOCUMENT, SCENE_TRAJECTORY


# ======================== Trajectory Extraction ========================

TRAJECTORY_EXTRACTION_PROMPT_EN = """You are a precise trace indexer. The input is an agent execution trajectory: an ordered sequence of steps, each with an action and the resulting observation/environment state.

## Knowledge Point Format
Each knowledge point MUST contain these four fields:
- **time**: The step anchor, written exactly as "Step N" (N = the step index this fact belongs to). For a fact spanning a range, write "Step A-B". NEVER leave this empty when a step index exists.
- **subject**: The acting agent or the concrete target of the action (element label, page/view name, field name). Never a pronoun.
- **fact**: A self-contained statement that MUST carry, whenever present in the source: the step index, the action verb (click / type / scroll / goto / hover / select / stop), the target element together with its bracketed reference id, the element's label text, and what the observation shows after the action.
- **entities_or_values**: All uniquely identifying tokens: every bracketed reference id copied VERBATIM including brackets (e.g. "[29575]"), element labels, page titles, URLs, field values, numbers, and the step index as "Step N".

## Mandatory Coverage (do NOT skip any)
1. **Every step**: each step index in the trajectory MUST produce at least one knowledge point. Do not skip steps, do not summarize consecutive steps into one.
2. **Action bindings**: the pairing of step index -> action -> target element id -> element label is the single most important fact type. Extract it for EVERY step that has an action.
3. **State of the view**: the page/view title, URL, or root node label observed at that step, when it changes or when it identifies the step.
4. **State transitions**: any value, message, count, or status that differs from the previous step, stated as "changed from X to Y".
5. **Repetition and loops**: if the same view, action, or element recurs at several steps, emit one knowledge point listing ALL step indices where it recurs.
6. **Task framing**: the overall task/goal statement, and the final outcome or stop condition.

## Critical Rules
- **Copy identifiers verbatim**: bracketed reference ids must appear exactly as in the source, brackets included. Never paraphrase, renumber, or drop them.
- **No cross-step merging**: one knowledge point must not describe two different step indices unless it is an explicit repetition/range fact (rule 5).
- **No generalization**: never write "clicked a link" — write which link, with its id and label.
- **Zero information loss**: every reference id, title, URL and numeric value in the source must appear in some knowledge point.

## Trajectory Content
{session_text}

## Output
Output all knowledge points as a JSON array (one or more per step; a long trajectory yields 30-80):
```json
[
  {{"time": "Step 0", "subject": "CATALOG link", "fact": "At Step 0 the agent clicked link '\\ue608 CATALOG' with reference id [96], and the observation then showed the Magento Admin dashboard with the catalog menu expanded", "entities_or_values": ["Step 0", "[96]", "CATALOG", "Magento Admin"]}},
  {{"time": "Step 12", "subject": "All Customers link", "fact": "At Step 12 the agent clicked link 'All Customers' with reference id [29575], navigating to the customer grid", "entities_or_values": ["Step 12", "[29575]", "All Customers"]}},
  {{"time": "Step 9-17", "subject": "RootWebArea title", "fact": "The RootWebArea title 'Customers / Customers / Magento Admin' was observed at Step 9, Step 13 and Step 17, indicating the agent returned to the same customer page three times", "entities_or_values": ["Step 9", "Step 13", "Step 17", "Customers / Customers / Magento Admin"]}},
  {{"time": "Step 14", "subject": "System Messages region", "fact": "At Step 14 the System Messages region reference id changed from [23778] to [38405] while the page title stayed the same", "entities_or_values": ["Step 14", "[23778]", "[38405]", "System Messages"]}}
]
```"""


TRAJECTORY_EXTRACTION_PROMPT_ZH = """你是一个精确的执行轨迹索引助手。输入是一段 Agent 执行轨迹：由若干有序步骤组成，每步包含一个动作和执行后的观测/环境状态。

## 知识点格式
每条知识点必须包含以下四个字段：
- **time**: 步骤锚点，严格写成 "Step N"（N 为该事实所属的步骤序号）。若事实跨越区间，写成 "Step A-B"。只要存在步骤序号就绝不能留空。
- **subject**: 执行主体或动作的具体目标（元素标签、页面/视图名、字段名）。不得使用代词。
- **fact**: 自包含的事实陈述，只要原文中存在就必须携带：步骤序号、动作类型（click / type / scroll / goto / hover / select / stop）、目标元素及其方括号引用 ID、元素的标签文本、以及动作后观测到的结果。
- **entities_or_values**: 所有唯一标识符：每一个方括号引用 ID 必须**原样逐字复制**（含方括号，如 "[29575]"）、元素标签、页面标题、URL、字段值、数值，以及 "Step N" 形式的步骤序号。

## 强制覆盖维度（不得遗漏）
1. **每一个步骤**：轨迹中的每个步骤序号都必须至少产出一条知识点。不得跳过步骤，不得把连续步骤合并成一条。
2. **动作绑定**：步骤序号 → 动作 → 目标元素 ID → 元素标签 这一组绑定关系是最重要的事实类型，每个有动作的步骤都必须提取。
3. **视图状态**：该步骤观测到的页面/视图标题、URL 或根节点标签，尤其在发生变化或可用于标识该步骤时。
4. **状态变迁**：与上一步相比发生变化的任何值、消息、计数或状态，写成"从 X 变为 Y"。
5. **重复与循环**：若同一视图、动作或元素在多个步骤重复出现，输出一条知识点并列出**全部**重复的步骤序号。
6. **任务框架**：整体任务/目标陈述，以及最终结果或终止条件。

## 关键规则
- **标识符逐字复制**：方括号引用 ID 必须与原文完全一致（含方括号），禁止改写、重新编号或省略。
- **禁止跨步合并**：一条知识点不得描述两个不同步骤序号，除非它是规则 5 的显式重复/区间事实。
- **禁止泛化**：不得写"点击了一个链接"，必须写明是哪个链接及其 ID 与标签。
- **信息零损耗**：原文中每一个引用 ID、标题、URL 和数值都必须出现在某条知识点中。

## 轨迹内容
{session_text}

## 输出
以 JSON 数组输出所有知识点（每步一条以上，长轨迹通常 30-80 条）：
```json
[
  {{"time": "Step 0", "subject": "CATALOG 链接", "fact": "在 Step 0，agent 点击了引用 ID 为 [96] 的链接 '\\ue608 CATALOG'，随后观测到 Magento Admin 仪表盘并展开了目录菜单", "entities_or_values": ["Step 0", "[96]", "CATALOG", "Magento Admin"]}},
  {{"time": "Step 12", "subject": "All Customers 链接", "fact": "在 Step 12，agent 点击了引用 ID 为 [29575] 的链接 'All Customers'，跳转到客户列表页", "entities_or_values": ["Step 12", "[29575]", "All Customers"]}},
  {{"time": "Step 9-17", "subject": "RootWebArea 标题", "fact": "标题为 'Customers / Customers / Magento Admin' 的 RootWebArea 在 Step 9、Step 13、Step 17 各出现一次，说明 agent 三次回到同一客户页面", "entities_or_values": ["Step 9", "Step 13", "Step 17", "Customers / Customers / Magento Admin"]}},
  {{"time": "Step 14", "subject": "System Messages 区域", "fact": "在 Step 14，System Messages 区域的引用 ID 从 [23778] 变为 [38405]，而页面标题保持不变", "entities_or_values": ["Step 14", "[23778]", "[38405]", "System Messages"]}}
]
```"""


# ======================== Document Extraction ========================

DOCUMENT_EXTRACTION_PROMPT_EN = """You are a precise knowledge point extraction assistant. Given a document or record, extract ALL structured knowledge points from it.

## Knowledge Point Format
Each knowledge point MUST contain these four fields:
- **time**: The absolute date the described event occurred, resolved from any relative reference in the text. Empty string if the text carries no time reference at all.
- **subject**: The entity the fact is about (person, organization, product, system, measurement). Never a pronoun.
- **fact**: A **high-density, self-contained statement**: "subject + action/state + object + ALL modifiers (time / place / quantity / manner / cause / result)". Retain every specific value from the source; generalize nothing.
- **entities_or_values**: All uniquely distinguishing identifiers: proper nouns, identifiers/codes, numeric values with units, and dates.

## Mandatory Coverage (do NOT skip any)
1. **Definitions and attributes**: what each entity is, its properties, categories, and identifiers.
2. **Quantitative facts**: every number, measurement, threshold, duration, cost, count — with its unit and what it measures.
3. **Temporal facts**: every dated or sequenced event, including ordering between events.
4. **Causal and conditional facts**: reasons, effects, prerequisites, "if X then Y" rules.
5. **Relations**: how entities relate to one another (part-of, owned-by, derived-from, compared-with).
6. **Changes and comparisons**: before/after values, deltas, trends, differences between entities.
7. **Procedures and rules**: ordered instructions, constraints, policies, exceptions.

## Critical Rules
- **Zero information loss**: every number, name, date, identifier and unit in the source MUST appear in some knowledge point.
- **One sentence, multiple KPs**: never merge distinct facts into one knowledge point.
- **Self-contained**: each fact must be understandable alone, with full names instead of pronouns or "it".
- **No vague language**: never use "etc.", "various", "some" in place of explicit enumeration.
- **Copy identifiers verbatim**: codes, ids, version strings and units exactly as written in the source.

## Document Content
{session_text}

## Output
Output all knowledge points as a JSON array (typically 20-60 per document):
```json
[
  {{"time": "March 2024", "subject": "Model X-7 sensor", "fact": "The Model X-7 sensor was rated for a 0-150 degC operating range with an accuracy of +/-0.3 degC, certified under standard IEC-60751 in March 2024", "entities_or_values": ["Model X-7", "0-150 degC", "+/-0.3 degC", "IEC-60751", "March 2024"]}},
  {{"time": "", "subject": "Retry policy", "fact": "The retry policy performs at most 3 attempts with exponential backoff starting at 200 ms, and aborts the request when the total elapsed time exceeds 5 seconds", "entities_or_values": ["3 attempts", "200 ms", "5 seconds", "exponential backoff"]}}
]
```"""


DOCUMENT_EXTRACTION_PROMPT_ZH = """你是一个精确的知识点提取助手。给定一篇文档或记录，从中提取所有结构化知识点。

## 知识点格式
每条知识点必须包含以下四个字段：
- **time**: 所述事件发生的绝对日期，需从文中任何相对时间引用推算得到。若文中完全没有时间引用则留空字符串。
- **subject**: 该事实所描述的实体（人物、机构、产品、系统、指标）。不得使用代词。
- **fact**: **高密度、自包含**的事实陈述："主体 + 动作/状态 + 对象 + 全部修饰语（时间/地点/数量/方式/原因/结果）"。保留原文中所有具体数值，不做任何概括。
- **entities_or_values**: 所有唯一区分性标识符：专有名词、编号/代码、带单位的数值、日期。

## 强制覆盖维度（不得遗漏）
1. **定义与属性**：每个实体是什么，其属性、类别与标识符。
2. **量化事实**：每一个数字、指标、阈值、时长、金额、计数，连同单位及其度量对象。
3. **时间事实**：每一个带日期或有先后顺序的事件，包括事件之间的次序关系。
4. **因果与条件**：原因、影响、前置条件、"若 X 则 Y"型规则。
5. **关系**：实体之间的关系（从属、归属、派生、对比）。
6. **变化与对比**：前后数值、差值、趋势、实体间差异。
7. **流程与规则**：有序步骤、约束、策略、例外情况。

## 关键规则
- **信息零损耗**：原文中每一个数字、名称、日期、标识符和单位都必须出现在某条知识点中。
- **一句多提**：不得将不同事实合并为一条知识点。
- **自包含**：每条 fact 必须可独立理解，使用全称而非代词或"它"。
- **禁止模糊表述**：不得用"等"、"若干"、"相关"替代明确列举。
- **标识符逐字复制**：代码、ID、版本号与单位必须与原文完全一致。

## 文档内容
{session_text}

## 输出
以 JSON 数组输出所有知识点（通常每篇 20-60 条）：
```json
[
  {{"time": "2024年3月", "subject": "X-7 型传感器", "fact": "X-7 型传感器的工作温区为 0-150 摄氏度，精度为正负 0.3 摄氏度，于 2024 年 3 月通过 IEC-60751 标准认证", "entities_or_values": ["X-7", "0-150摄氏度", "±0.3摄氏度", "IEC-60751", "2024年3月"]}},
  {{"time": "", "subject": "重试策略", "fact": "重试策略最多执行 3 次尝试，采用起始 200 毫秒的指数退避，当累计耗时超过 5 秒时终止请求", "entities_or_values": ["3次", "200毫秒", "5秒", "指数退避"]}}
]
```"""


# ======================== Scene-specific FQ dimensions ========================

TRAJECTORY_DIMENSION_SPEC_EN = """## Content Dimensions (8 query angles for execution traces)
| Dimension | Query Focus |
|-----------|-------------|
| step_action | What action was taken at step N / at which step was action A taken |
| element_ref | Which element reference id was used for a given step or label |
| state_at_step | What the view/title/value was at a given step |
| state_change | What changed between two steps, from which value to which |
| repetition | Which step indices share the same view, action, or element |
| navigation_path | Order of steps/pages traversed between two points |
| outcome | Final result, stop condition, whether the task succeeded |
| natural_recall | Short (5-15 words) natural recall of a step or element |
"""

TRAJECTORY_DIMENSION_SPEC_ZH = """## 内容维度（针对执行轨迹的 8 种查询角度）
| 维度 | 查询重点 |
|------|----------|
| step_action | 第 N 步执行了什么动作 / 动作 A 发生在第几步 |
| element_ref | 某步骤或某标签对应的元素引用 ID 是什么 |
| state_at_step | 某步骤时视图/标题/取值是什么 |
| state_change | 两步之间发生了什么变化、从什么值变成什么值 |
| repetition | 哪些步骤序号共享同一视图、动作或元素 |
| navigation_path | 两点之间经过的步骤/页面顺序 |
| outcome | 最终结果、终止条件、任务是否成功 |
| natural_recall | 对某步骤或元素的短句（5-15 字）自然回忆 |
"""

TRAJECTORY_CONSTRAINTS_EN = """## Scene-specific Constraints (execution trace)
- This scene OVERRIDES core constraint 3: perspective mixing does not apply. Write every query in neutral third-person about the trace ("Which element was clicked at Step 12?").
- Every query that concerns a step MUST name the step index explicitly ("at Step 12", "between Step 6 and Step 22").
- At least 40% of queries must ask for a reference id, a label, or a title observed at a specific step.
- At least 2 queries must span a step RANGE or ask which steps repeat the same state.
- Copy reference ids verbatim with brackets when they appear in a query.
"""

TRAJECTORY_CONSTRAINTS_ZH = """## 场景专属约束（执行轨迹）
- 本场景**覆盖核心约束 3**：不适用人称视角混合要求。所有查询一律使用中性第三人称描述轨迹（如"Step 12 点击的是哪个元素？"）。
- 任何涉及步骤的查询都必须显式写出步骤序号（如"在 Step 12"、"Step 6 到 Step 22 之间"）。
- 至少 40% 的查询应当询问某个具体步骤上的引用 ID、元素标签或页面标题。
- 至少 2 条查询需跨越步骤**区间**，或询问哪些步骤重复了同一状态。
- 查询中出现引用 ID 时必须连同方括号逐字复制。
"""

DOCUMENT_DIMENSION_SPEC_EN = """## Content Dimensions (8 query angles for documents)
| Dimension | Query Focus |
|-----------|-------------|
| entity_fact | Who/What/How much precise factual queries |
| definition | What is X / how is X categorized |
| quantitative | Specific numbers, thresholds, units, limits |
| temporal | When/sequence/time range queries |
| causal | Why/root cause/condition/consequence queries |
| relation | How two entities relate or compare |
| procedure | Ordered steps, rules, constraints, exceptions |
| natural_recall | Short (5-15 words), natural recall queries |
"""

DOCUMENT_DIMENSION_SPEC_ZH = """## 内容维度（针对文档的 8 种查询角度）
| 维度 | 查询重点 |
|------|----------|
| entity_fact | Who/What/How much 精确事实查询 |
| definition | X 是什么 / X 属于哪一类 |
| quantitative | 具体数值、阈值、单位、上限 |
| temporal | When/顺序/时间范围查询 |
| causal | Why/根因/条件/后果查询 |
| relation | 两个实体之间的关系或对比 |
| procedure | 有序步骤、规则、约束、例外 |
| natural_recall | 短句（5-15 字）自然回忆查询 |
"""


_EXTRACTION_PROMPTS = {
    SCENE_TRAJECTORY: (TRAJECTORY_EXTRACTION_PROMPT_ZH, TRAJECTORY_EXTRACTION_PROMPT_EN),
    SCENE_DOCUMENT: (DOCUMENT_EXTRACTION_PROMPT_ZH, DOCUMENT_EXTRACTION_PROMPT_EN),
}

_DIMENSION_SPECS = {
    SCENE_TRAJECTORY: (TRAJECTORY_DIMENSION_SPEC_ZH, TRAJECTORY_DIMENSION_SPEC_EN),
    SCENE_DOCUMENT: (DOCUMENT_DIMENSION_SPEC_ZH, DOCUMENT_DIMENSION_SPEC_EN),
}

_EXTRA_CONSTRAINTS = {
    SCENE_TRAJECTORY: (TRAJECTORY_CONSTRAINTS_ZH, TRAJECTORY_CONSTRAINTS_EN),
}


def _pick(table: dict, scene: str, language: str) -> str | None:
    pair = table.get(scene)
    if pair is None:
        return None
    return pair[0] if language == "zh" else pair[1]


def get_extraction_prompt(scene: str, language: str) -> str | None:
    """Scene-specific extraction template, or None for the dialogue default."""
    return _pick(_EXTRACTION_PROMPTS, scene, language)


def get_scene_dimension_spec(scene: str, language: str) -> str | None:
    """Scene-specific FQ dimension table, or None for the dialogue default."""
    return _pick(_DIMENSION_SPECS, scene, language)


def get_scene_constraints(scene: str, language: str) -> str:
    """Extra FQ constraints appended for scenes that need them."""
    return _pick(_EXTRA_CONSTRAINTS, scene, language) or ""


__all__ = [
    "SCENE_DIALOGUE",
    "SCENE_DOCUMENT",
    "SCENE_TRAJECTORY",
    "get_extraction_prompt",
    "get_scene_dimension_spec",
    "get_scene_constraints",
]
