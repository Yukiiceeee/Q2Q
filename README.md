# Q2Q: Query-to-Query Agent Memory System

基于假想查询生成与双路联合检索的 Agent 记忆系统。

## 核心思路

- **记忆构建阶段**：对每轮会话生成假想查询（Fake Queries），构建 query→memory 索引
- **查询问答阶段**：将用户查询分解为子查询，与假想查询做 Q2Q + Q2C 双路相似度检索

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
# 记忆存储
python main.py memorize --file data/session.txt

# 查询检索
python main.py query "患者的血糖控制情况如何？"

# 运行 MedMemoryBench 评测
python main.py evaluate --max-sessions 10

# 查看统计
python main.py stats

# 清空记忆
python main.py clear
```

## 配置

通过 `.env` 文件配置，关键参数：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEFAULT_LLM_MODEL` | LLM 模型 | gpt-4o-mini |
| `EMBEDDING_PROVIDER` | 嵌入模式 (local/openai) | local |
| `DEFAULT_EMBEDDING_MODEL` | 嵌入模型路径 | - |
| `STORAGE_BACKEND` | 存储后端 (chromadb/json) | chromadb |
| `RETRIEVAL_ALPHA` | Q2Q 权重 α | 0.7 |
| `NUM_FAKE_QUERIES` | 假想查询数量 | 5 |
| `PROMPT_LANGUAGE` | 提示词语言 (zh/en) | zh |

## 项目结构

```
Q2Q/
├── agent.py                 # 顶层编排 (memorize/query 入口)
├── main.py                  # CLI 入口
├── src/
│   ├── config.py            # 全局配置
│   ├── schemas/             # 数据模型
│   ├── prompts/             # 中英文提示词模板
│   ├── embedding/           # 嵌入计算 (本地/API)
│   ├── memory/              # 记忆构建 (假想查询生成)
│   ├── retrieval/           # 查询检索 (分解+双路检索)
│   ├── storage/             # 存储层 (ChromaDB/JSON)
│   └── utils/               # LLM客户端/日志/分词器
├── tests/
│   ├── test_q2q.py          # 单元测试
│   └── evaluate.py          # MedMemoryBench 评测脚本
├── data/                    # 数据文件
├── logs/                    # 日志输出
└── outputs/                 # 评测结果输出
```

## 测试

```bash
python tests/test_q2q.py
```
