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
| `EMBEDDING_DEVICE` | 嵌入设备 | cpu |
| `COMPRESSOR_MODEL` | 困惑度压缩模型路径 | - |
| `PROPOSITION_MODE` | 命题提取模式 (llm/nlp) | llm |
| `STORAGE_CHROMADB_PATH` | ChromaDB 持久化目录 | data/chromadb |
| `FQ_STYLE` | 假查询表述样式 (formal/colloquial/keyword/elliptical) | formal |
| `FQ_ROLE` | 假查询角色视角 (user/otherside) | user |
| `LOG_LEVEL` | 日志级别 | INFO |

> 场景路由（dialogue / trajectory / document）由 `src/preprocessing/scene_detector.py`
> 在记忆构建时根据文本结构自动判定，无需配置。
> 注意：双路检索中 Q2Q 与 Q2C 结果直接合并，不存在 α 加权。

## 项目结构

```
Q2Q/
├── agent.py                 # 顶层编排 (memorize/query 入口)
├── main.py                  # CLI 入口
├── src/
│   ├── config.py            # 全局配置
│   ├── schemas/             # 数据模型
│   ├── prompts/             # 中英文提示词模板（含场景路由模板）
│   ├── embedding/           # 嵌入计算 (本地/API)
│   ├── memory/              # 记忆构建 (假想查询生成 + DAG 图管理)
│   ├── retrieval/           # 查询检索 (分解+双路检索)
│   ├── storage/             # 存储层 (ChromaDB)
│   └── utils/               # LLM客户端/日志/分词器
├── experiment/              # Motivation 分析实验（生成/嵌入/存储/实验执行）
├── scripts/                 # 启动脚本
├── data/                    # 数据文件
├── logs/                    # 日志输出
└── outputs/                 # 评测结果输出
```

> 评测流程由外部 harness（MedMemoryBench `scripts/run_eval.sh`）驱动，本仓库不再内置评测入口。
