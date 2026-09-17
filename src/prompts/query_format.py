QUERY_FORMAT_SPEC_ZH = """## 查询格式规范
每条查询必须遵循以下规则：
1. **结构要求**：遵循"时间+主体+事件+结果"的结构组织，尽可能包含这四个要素。
   - 时间：事件发生的时间/时间段（若已知）
   - 主体：涉及的人物/实体
   - 事件：发生的动作/状态/关系
   - 结果：期望查询的具体数值/状态/结论
2. 简短直接的问句，8-30 字，只问一个事实。
3. 以疑问词开头（什么/如何/为什么/何时/哪里/哪个/是否/多少）。
4. 包含具体实体名称和关键细节，能有一个明确的答案。
5. 不得引用"对话"或"会话"——作为独立问题。
6. 语言：与原始内容语言一致。

"""

QUERY_FORMAT_SPEC_EN = """## Query Format Specification
Each query MUST follow these rules:
1. **Structure Requirement**: Follow the "Time + Subject + Event + Result" structure, including as many of these four elements as possible.
   - Time: when the event occurred or the time range (if known)
   - Subject: the person/entity involved
   - Event: the action/state/relationship that occurred
   - Result: the specific value/status/conclusion being queried
2. Short, direct question form, 8-30 words, asking about ONE fact.
3. Begin with an interrogative word (What/How/Why/When/Where/Which/Does/Is/How much).
4. Include specific entity names and key details for a concrete answer.
5. No references to "the conversation" or "the session" — standalone question.
6. Language: Match the language of the source content.

"""

# FQ-specific format spec: allows longer, more detailed queries
FQ_FORMAT_SPEC_ZH = """## 假想查询格式规范
每条假想查询必须遵循以下规则：
1. **结构要求**：遵循"时间+主体+事件+结果"的结构组织，包含具体细节。
   - 时间：事件发生的具体时间/日期/时间段
   - 主体：涉及的具体人物姓名/机构名称/实体
   - 事件：发生的具体动作/状态/关系（含关键数值、剂量、频率等）
   - 结果：期望查询的具体数值/状态/结论
2. 简洁精准的问句，8-40 字。包含关键信息但避免过度堆砌细节。
3. 以疑问词开头（什么/如何/为什么/何时/哪里/哪个/是否/多少）。
4. 必须包含具体实体名称和能定位答案的关键细节。
5. 不得引用"对话"或"会话"——作为独立问题。
6. 语言：与原始内容语言一致。
7. **自然查询模拟**：约30%的查询应模拟真实用户的简短自然提问（8-15字），类似真实记忆回忆场景。

**示例对比**：
- 差（太笼统）：「患者的血压是多少？」
- 好（具体）：「张某3月15日复诊时收缩压是多少？」
- 好（自然简短）：「张某的血压多少？」

"""

FQ_FORMAT_SPEC_EN = """## Fake Query Format Specification
Each hypothetical query MUST follow these rules:
1. **Structure Requirement**: Follow the "Time + Subject + Event + Result" structure, including key specific details.
   - Time: the specific time/date/period of the event
   - Subject: specific person names/organization names/entities involved
   - Event: specific action/state/relationship (including key values, dosages, frequencies, etc.)
   - Result: the specific value/status/conclusion being queried
2. Concise and precise question form, 8-40 words. Include key information but avoid over-stuffing details.
3. Begin with an interrogative word (What/How/Why/When/Where/Which/Does/Is/How much).
4. MUST include specific entity names and key details that can locate the answer.
5. No references to "the conversation" or "the session" — standalone question.
6. Language: Match the language of the source content.
7. **Natural query simulation**: ~30% of queries should simulate real users' short, natural questions (8-15 words), resembling real memory recall scenarios.

**Example Comparison**:
- Bad (too vague): "What was the patient's blood pressure?"
- Good (specific): "What was Mr. Zhang's systolic reading at the March 15 follow-up?"
- Good (natural & short): "When did Sarah visit the museum?"

"""


def get_query_format_spec(language: str = "zh") -> str:
    if language == "zh":
        return QUERY_FORMAT_SPEC_ZH
    return QUERY_FORMAT_SPEC_EN


def get_fq_format_spec(language: str = "zh") -> str:
    if language == "zh":
        return FQ_FORMAT_SPEC_ZH
    return FQ_FORMAT_SPEC_EN


QUERY_FORMAT_SPEC = QUERY_FORMAT_SPEC_EN
