QUERY_FORMAT_SPEC_EN = """## Query Format Specification
Each query MUST follow these rules:
1. Written as a direct, self-contained question.
2. Begin with an interrogative word (What/How/Why/When/Where/Which/Can/Does/Is).
3. Be specific enough to have a concrete answer.
4. Contain no references to "the conversation" or "the session" — treat it as a standalone question.
5. Length: 10-40 words per query.
6. Language: Match the language of the source content.
"""

QUERY_FORMAT_SPEC_ZH = """## 查询格式规范
每条查询必须遵循以下规则：
1. 以直接、独立的问句形式书写。
2. 以疑问词开头（什么/如何/为什么/何时/哪里/哪个/是否/能否）。
3. 足够具体，能有一个明确的答案。
4. 不得引用"对话"或"会话"——将其视为一个独立的问题。
5. 长度：每条查询 10-50 字。
6. 语言：与原始内容语言一致。
"""


def get_query_format_spec(language: str = "zh") -> str:
    if language == "zh":
        return QUERY_FORMAT_SPEC_ZH
    return QUERY_FORMAT_SPEC_EN


# Backward compatibility
QUERY_FORMAT_SPEC = QUERY_FORMAT_SPEC_EN
