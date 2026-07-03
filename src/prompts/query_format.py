QUERY_FORMAT_SPEC_ZH = """## 查询格式规范
每条查询必须遵循以下规则：
1. 采用结构化问句形式，尽量涵盖以下要素：时间 + 主体/角色 + 事件/行为 + 结果/状态。
2. 以疑问词开头（什么/如何/为什么/何时/哪里/哪个/是否/能否）。
3. 足够具体，包含关键细节，能有一个明确的答案。
4. 不得引用"对话"或"会话"——将其视为一个独立的问题。
5. 长度：每条查询 15-80 字。
6. 语言：与原始内容语言一致。

"""

QUERY_FORMAT_SPEC_EN = """## Query Format Specification
Each query MUST follow these rules:
1. Use a structured question form covering: time + subject/role + event/action + result/state.
2. Begin with an interrogative word (What/How/Why/When/Where/Which/Can/Does/Is).
3. Be specific enough to have a concrete answer, including key details (specific names, values, time points).
4. Contain no references to "the conversation" or "the session" — treat it as a standalone question.
5. Length: 15-80 words per query.
6. Language: Match the language of the source content.

"""


def get_query_format_spec(language: str = "zh") -> str:
    if language == "zh":
        return QUERY_FORMAT_SPEC_ZH
    return QUERY_FORMAT_SPEC_EN


QUERY_FORMAT_SPEC = QUERY_FORMAT_SPEC_EN
