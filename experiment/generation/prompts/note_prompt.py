"""Prompt templates for generating memory note variants.

Three variants targeting different memory construction paradigms:
- Proposition (Q2P): Atomic fact extraction, inspired by Mem0 / HippoRAG
- Note (Q2N): Structured Zettelkasten notes, inspired by A-Mem
- Reflection (Q2R): High-level insights, inspired by Generative Agents
"""

# ---------------------------------------------------------------------------
#  Q2P — Atomic Propositions  (Mem0 / HippoRAG style)
# ---------------------------------------------------------------------------

PROPOSITION_PROMPT_EN = """You are a memory extraction system. Extract atomic propositions from the following conversation.

Each proposition must:
1. Be a single, self-contained declarative sentence expressing exactly one fact
2. Use full proper names instead of pronouns (replace "he/she/they" with the person's name)
3. Preserve all specific details: names, dates, locations, numbers, quantities
4. Be independently understandable without reading the original conversation
5. Cover ALL factual information, including: personal attributes, events, temporal facts, preferences, relationships, plans, emotional states, opinions

Output ONLY a JSON array of strings. Generate 15-25 propositions.

Conversation:
{session_text}"""

PROPOSITION_PROMPT_ZH = """你是一个记忆提取系统。请从以下对话中提取原子命题。

每条命题必须：
1. 是一个独立的、自包含的陈述句，只表达一个事实
2. 使用完整的人名而非代词（将"他/她/他们"替换为具体姓名）
3. 保留所有具体细节：姓名、日期、地点、数字、数量
4. 无需阅读原始对话即可独立理解
5. 覆盖所有事实信息，包括：个人属性、事件、时间事实、偏好、关系、计划、情感状态、观点

仅输出一个 JSON 字符串数组。生成 15-25 条命题。

对话内容：
{session_text}"""

# ---------------------------------------------------------------------------
#  Q2N — Structured Notes  (A-Mem / Zettelkasten style)
# ---------------------------------------------------------------------------

NOTE_PROMPT_EN = """You are a memory note system using the Zettelkasten method. Create structured memory notes from the following conversation.

Each note should capture one coherent topic or theme. For each note, output a JSON object with these fields:
- "title": A concise descriptive title (5-10 words)
- "key_insight": The core takeaway or insight (1-2 sentences)
- "content": Detailed factual summary covering who, what, when, where, why (2-4 sentences)
- "tags": List of 3-5 keyword tags for categorization

Requirements:
- Each note should be self-contained and independently understandable
- Preserve specific names, dates, numbers, and details
- Cover different aspects of the conversation (don't repeat information across notes)
- Generate 5-10 notes depending on conversation richness

Output ONLY a JSON array of note objects.

Conversation:
{session_text}"""

NOTE_PROMPT_ZH = """你是一个使用 Zettelkasten 方法的记忆笔记系统。请从以下对话中创建结构化记忆笔记。

每条笔记应捕捉一个连贯的话题或主题。每条笔记输出一个 JSON 对象，包含以下字段：
- "title": 简洁描述性标题（5-10 个词）
- "key_insight": 核心要点或洞察（1-2 句话）
- "content": 详细事实摘要，涵盖谁、什么、何时、何地、为什么（2-4 句话）
- "tags": 3-5 个关键词标签列表

要求：
- 每条笔记应自包含，可独立理解
- 保留具体的人名、日期、数字和细节
- 覆盖对话的不同方面（各笔记之间不重复信息）
- 根据对话丰富程度生成 5-10 条笔记

仅输出一个 JSON 对象数组。

对话内容：
{session_text}"""

# ---------------------------------------------------------------------------
#  Q2R — Reflections  (Generative Agents style)
# ---------------------------------------------------------------------------

REFLECTION_PROMPT_EN = """You are a reflection module in a long-term memory system. Based on the following conversation, generate high-level reflections and insights.

Each reflection should:
1. Abstract beyond surface-level facts to identify patterns, personality traits, relationship dynamics, or behavioral tendencies
2. Capture implications that might be relevant to future interactions
3. Identify emotional undercurrents, motivations, or values that drive the speakers' behavior
4. Note significant state changes, turning points, or evolving perspectives
5. Be expressed as a single declarative insight statement (1-2 sentences)

Types of reflections to generate:
- Character insights: "X tends to...", "X values...", "X is going through..."
- Relationship dynamics: "The relationship between X and Y is characterized by..."
- Behavioral patterns: "When X encounters..., they typically..."
- State transitions: "X's attitude toward... has shifted from... to..."
- Future implications: "Based on..., X is likely to..."

Generate 8-12 reflections. Output ONLY a JSON array of strings.

Conversation:
{session_text}"""

REFLECTION_PROMPT_ZH = """你是长期记忆系统中的反思模块。请根据以下对话生成高层次的反思和洞察。

每条反思应：
1. 超越表面事实，识别行为模式、性格特征、关系动态或行为倾向
2. 捕捉可能与未来交互相关的含义
3. 识别驱动说话者行为的情感暗流、动机或价值观
4. 注意重要的状态变化、转折点或不断演变的观点
5. 以一个陈述性洞察语句表达（1-2 句话）

应生成的反思类型：
- 性格洞察："X 倾向于……"、"X 重视……"、"X 正在经历……"
- 关系动态："X 和 Y 之间的关系特点是……"
- 行为模式："当 X 遇到……时，通常会……"
- 状态转变："X 对……的态度已从……转变为……"
- 未来影响："基于……，X 可能会……"

生成 8-12 条反思。仅输出一个 JSON 字符串数组。

对话内容：
{session_text}"""

# ---------------------------------------------------------------------------
#  Constants & Builders
# ---------------------------------------------------------------------------

NOTE_VARIANT_STYLES = ["proposition", "note", "reflection"]

_PROMPT_MAP = {
    "proposition": {"en": PROPOSITION_PROMPT_EN, "zh": PROPOSITION_PROMPT_ZH},
    "note": {"en": NOTE_PROMPT_EN, "zh": NOTE_PROMPT_ZH},
    "reflection": {"en": REFLECTION_PROMPT_EN, "zh": REFLECTION_PROMPT_ZH},
}


def build_proposition_prompt(session_text: str, language: str = "en") -> str:
    tpl = PROPOSITION_PROMPT_ZH if language == "zh" else PROPOSITION_PROMPT_EN
    return tpl.format(session_text=session_text[:3000])


def build_note_prompt(session_text: str, language: str = "en") -> str:
    tpl = NOTE_PROMPT_ZH if language == "zh" else NOTE_PROMPT_EN
    return tpl.format(session_text=session_text[:3000])


def build_reflection_prompt(session_text: str, language: str = "en") -> str:
    tpl = REFLECTION_PROMPT_ZH if language == "zh" else REFLECTION_PROMPT_EN
    return tpl.format(session_text=session_text[:3000])


def build_variant_prompt(session_text: str, style: str, language: str = "en") -> str:
    lang_key = "zh" if language == "zh" else "en"
    tpl = _PROMPT_MAP[style][lang_key]
    return tpl.format(session_text=session_text[:3000])
