# Q2Q: Query-to-Query Agent Memory System

基于假想查询生成与双路联合检索的 Agent 记忆系统。

## 核心思路

- **记忆构建阶段**：对每轮会话生成假想查询（Fake Queries），构建 query→memory 索引
- **查询问答阶段**：将用户查询分解为子查询，与假想查询做 Q2Q + Q2C 双路相似度检索

## 安装

```bash
# 核心运行时依赖
pip install -r requirements.txt

# 仅跑 experiment/ 下的分析实验时额外需要
pip install -r requirements-experiment.txt

# 配置环境变量
cp .env.example .env
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

## 资产路径约定

仓库内不写死任何绝对路径。模型与数据集的根目录由两个环境变量决定，默认指向
项目内，可改指向共享缓存；配置文件里只写相对于这两个根的名字或路径，解析逻辑
统一收敛在 `src/paths.py`。

| 变量 | 作用 | 默认值 |
|------|------|--------|
| `MODELS_DIR` | 本地模型权重根目录 | `<项目根>/models` |
| `DATA_DIR` | 数据集根目录 | `<项目根>/data` |

例如 `experiment/configs/base.yaml` 里的 `model_name: "bge-large-en-v1.5"`
会解析为 `$MODELS_DIR/bge-large-en-v1.5`。已是绝对路径的值原样透传。

## 数据集

**本仓库不分发任何数据集。** LoCoMo 采用 CC BY-NC 4.0，其 NonCommercial 限制
会传递到由它生成的派生物（假想查询、命题、嵌入等），因此 `experiment/store/`
作为本地构建产物被 gitignore。各数据集的获取方式、预期路径与引用格式见
[DATASETS.md](DATASETS.md)。

## 配置

通过 `.env` 文件配置，完整变量清单见 `.env.example`。关键参数：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | LLM 提供方 (openai/anthropic/azure) | openai |
| `DEFAULT_LLM_MODEL` | LLM 模型 | gpt-4o-mini |
| `OPENAI_API_KEY` | LLM / 嵌入 API 密钥 | - |
| `OPENAI_BASE_URL` | API 端点（兼容网关时填写） | - |
| `MODELS_DIR` | 本地模型权重根目录 | `<项目根>/models` |
| `DATA_DIR` | 数据集根目录 | `<项目根>/data` |
| `EMBEDDING_PROVIDER` | 嵌入模式 (local/openai) | local |
| `DEFAULT_EMBEDDING_MODEL` | 嵌入模型名（相对 `MODELS_DIR`） | - |
| `EMBEDDING_DEVICE` | 嵌入设备 | cpu |
| `PROPOSITION_MODE` | 命题提取模式 (llm/nlp) | llm |
| `COMPRESSOR_MODEL` | 困惑度压缩模型名 | Qwen2.5-0.5B-Instruct |
| `NLP_INFERENCE_MODEL` | NLP 模式推理模型名 | flan-t5-large |
| `STORAGE_CHROMADB_PATH` | ChromaDB 持久化目录 | data/chromadb |
| `FQ_STYLE` | 假查询表述样式 (formal/colloquial/keyword/elliptical) | formal |
| `FQ_ROLE` | 假查询角色视角 (user/otherside) | user |
| `Q2Q_ANCHOR_WEIGHT` | 锚点查询加权系数 | - |
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
│   ├── paths.py             # 模型/数据集路径解析（项目根相对）
│   ├── schemas/             # 数据模型
│   ├── prompts/             # 中英文提示词模板（含场景路由模板）
│   ├── embedding/           # 嵌入计算 (本地/API)
│   ├── memory/              # 记忆构建 (假想查询生成 + DAG 图管理)
│   ├── retrieval/           # 查询检索 (分解+双路检索)
│   ├── storage/             # 存储层 (ChromaDB)
│   └── utils/               # LLM客户端/日志/分词器
├── experiment/              # Motivation 分析实验（生成/嵌入/存储/实验执行）
├── scripts/                 # 启动脚本
├── models/                  # 本地模型权重（不入库）
├── data/                    # 数据集（不入库）
├── logs/                    # 日志输出
└── outputs/                 # 评测结果输出
```

> 评测流程由外部 harness（MedMemoryBench `scripts/run_eval.sh`）驱动，本仓库不再内置评测入口。

## 实验产物

`experiment/` 下的 `store/`、`results/`、`figs/` 均为运行时产出，不随仓库分发：

```bash
python experiment/run.py          # 重新生成
open experiment/web/index.html    # 查看结果页
```

## 许可证

代码采用 [MIT](LICENSE)。第三方数据集各自保留其原始许可，详见
[DATASETS.md](DATASETS.md)——MIT 不覆盖、也无法覆盖这些语料。
