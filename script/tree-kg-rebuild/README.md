# Tree-KG Rebuild

本项目复现论文 **Tree-KG: An Expandable Knowledge Graph Construction Framework for Knowledge-intensive Domains** (Niu 等, ACL 2025) 的图谱构建流程，参考其官方仓库 [thu-pacman/Tree-KG](https://github.com/thu-pacman/Tree-KG) 提供的输入接口：用户上传 **一份 PDF + 一份用户配置 JSON**，由 LLM API 自动完成知识图谱构建。

> 与官方实现的差别：官方仓库目前只暴露了 HTTP 任务接口 (`submit_task` / `task_status` / `task_result`)，**没有放出代码**。这里基于论文方法论独立复现，全部使用本地脚本 + 用户自行配置的 LLM/Embedding API 完成。

## 功能特性

- **PDF → TOC → KG 骨架**：基于用户配置的页码区间 + 分级正则解析章节层级，构建 *tree-like hierarchical graph*（论文 §3.1.1）。
- **Phase 1 初始构建**（论文 §3.2）：自下而上摘要、实体抽取、章节/实体关系抽取。
- **Phase 2 迭代扩展**（论文 §3.3）：上下文卷积 conv、实体聚合 aggr、节点嵌入 embed、实体去重 dedup、两阶段边预测 pred。
- **每个环节独立配置 API**：可以让 `summary`、`extract`、`conv`、`aggr`、`dedup`、`pred`、`embed` 分别走不同的服务商/Key/模型。
- **LLM 调用统一封装**：上层不依赖 `openai` 模块，便于将来切换 SDK。
- **运行反馈**：彩色等级前缀 + tqdm 进度条 + 阶段统计 + 中间结果 checkpoint。
- **健壮异常处理**：网络抖动自动指数退避；鉴权失败/欠费/限频会**暂停程序**，等待用户处理后回车继续。

## 项目结构

```
tree-kg-rebuild/
├── README.md
├── requirements.txt
├── configs/
│   ├── user_config.example.json   # 用户侧（PDF/页码/正则）
│   └── api_config.example.json    # LLM 接入（每环节单独配置）
└── tree_kg/
    ├── __init__.py
    ├── __main__.py                # python -m tree_kg ...
    ├── main.py                    # CLI 入口
    ├── config.py                  # 配置加载
    ├── llm.py                     # LLM/Embedding 调用统一封装
    ├── prompts.py                 # 各阶段 prompt 模板
    ├── graph.py                   # KG 数据结构
    ├── pdf_parse.py               # PDF + 正则切分
    ├── phase1.py                  # 初始构建
    ├── phase2.py                  # 迭代扩展
    └── logger.py                  # 控制台输出 + 暂停工具
```

## 安装

> 建议 Python ≥ 3.10。

```bash
pip install -r requirements.txt
```

依赖：`openai`、`pymupdf`、`numpy`、`scikit-learn`、`networkx`、`tqdm`。

## 使用步骤

### 1. 准备用户配置 `user_config.json`

字段语义与官方文档完全一致（页码均为 PDF 文件页，1-based）：

```json
{
    "course_name": "Physics",
    "material_name": "Electromagnetic_Optics_Quantum_Physics.pdf",
    "book_start_page": 1,
    "book_end_page": 507,
    "cover_start_page": 1,
    "cover_end_page": 1,
    "toc_start_page": 2,
    "toc_end_page": 9,
    "text_start_page": 10,
    "text_end_page": 479,
    "appendix_start_page": 480,
    "appendix_end_page": 507,
    "toc_max_level": 3,
    "toc_re_expression": [
        "(第\\d+篇.*\\n\\n)",
        "(第\\d+章.*\\n\\n|.*物理趣闻.*\\n\\n|.*元素周期表.*\\n\\n|.*数值表.*\\n\\n|.*部分习题答案.*\\n\\n|.*索引.*\\n\\n)",
        "(\\*?\\d+\\.\\d+.*\\n\\n|.+?\\n\\n)"
    ]
}
```

> `toc_re_expression` 的每一个元素都必须包含一个**捕获组**（匹配标题文本）。脚本会按层级顺序对正文递归切分。

### 2. 准备 API 配置 `api_config.json`

支持「定义别名 + 各环节引用」的写法，覆写部分字段也可以：

```json
{
    "default": {
        "type": "chat",
        "api_key": "sk-xxx",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat"
    },
    "default_embedding": {
        "type": "embedding",
        "api_key": "sk-yyy",
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "BAAI/bge-m3",
        "batch_size": 16
    },
    "stages": {
        "summary": { "use": "default" },
        "extract": { "use": "default", "temperature": 0.0 },
        "conv":    { "use": "default" },
        "aggr":    { "use": "default" },
        "dedup":   { "use": "default" },
        "pred":    { "use": "default" },
        "embed":   { "use": "default_embedding" }
    },
    "runtime": {
        "checkpoint_dir": "./output/checkpoints",
        "output_dir": "./output",
        "dedup_threshold": 0.55,
        "edge_pred_strength_threshold": 6,
        "edge_pred_extra_edges": 1000,
        "edge_pred_alpha": 0.6,
        "edge_pred_beta_stage1": 0.0,
        "edge_pred_gamma_stage1": 0.4,
        "edge_pred_beta_stage2": 0.3,
        "edge_pred_gamma_stage2": 0.1
    }
}
```

支持的 stage 名：`summary / extract / conv / aggr / dedup / pred / embed`。
若想给 `extract` 单独换一家：

```json
"extract": {
    "type": "chat",
    "api_key": "sk-zzz",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini"
}
```

### 3. 运行

```bash
# 完整流程
python -m tree_kg \
    --pdf "/path/to/textbook.pdf" \
    --user-config configs/user_config.json \
    --api-config configs/api_config.json \
    --output-dir ./output

# 仅初始构建
python -m tree_kg --pdf ... --user-config ... --api-config ... --skip-phase2

# 从之前的 checkpoint 继续，跳过 Phase 1
python -m tree_kg --resume ./output/checkpoints/phase1_done.json \
    --pdf ... --user-config ... --api-config ...
```

### 4. 输出

- `output/kg_<course>.json` 最终图谱（节点 + 边）。
- `output/checkpoints/phase1_*.json` / `phase2_*.json` 中间快照，用于恢复或调试。

## 异常处理 & 暂停-继续

`tree_kg/llm.py` 中对 OpenAI 兼容 API 的常见错误进行了分类：

| 错误类型 | 行为 |
| --- | --- |
| 网络断开 / 超时 / 5xx | 指数退避自动重试，超过最大次数后**暂停**等待回车 |
| 401 鉴权失败 | **暂停** + 提示检查 api_key，回车后再次尝试 |
| 402 / 余额不足 / 403 | **暂停** + 提示充值或更换 Key，回车后再次尝试 |
| 429 限频 | **暂停** + 提示等待，回车后再次尝试 |
| 其他未知异常 | 提示后**暂停**，回车再次尝试 |

也就是说，运行中只要遇到 API 端可恢复的故障，**程序不会崩溃**，会停在那里等你处理好（充值/恢复网络/换 Key/等待限频结束），然后按回车即可继续。

## 数据结构 cheatsheet

节点 `Node`：

```text
id, name, layer, depth, description, raw_text, aliases, role, embedding, meta
```

边 `Edge`：

```text
src, dst, kind (vertical|horizontal), category, relation, description, strength, meta
```

层与边类别（论文 §3.1.2）：

| 层 | 含义 |
| --- | --- |
| `book` | 教材根 |
| `section` | 中间 TOC（篇/章/节...） |
| `subsection` | 叶子 TOC（含正文） |
| `core_entity` / `non_core_entity` | 实体两层 |

| 边类别 | 类型 |
| --- | --- |
| `has_subsection` | 垂直 (TOC) |
| `has_entity` | 垂直 (subsection→entity) |
| `has_subordinate` | 垂直 (core→non-core) |
| `section_related` | 水平 (同层 TOC) |
| `entity_related` | 水平 (同层实体) |

## 已知限制

- 论文中 `merge`（结构整合，新增材料合并）暂未实现，等待第二版迭代。
- TOC 正则切分的健壮性高度依赖于教材排版与 PDF 提取质量；请先用 `--skip-phase2` 跑一次，看 stdout 中的「大纲预览」是否合理，再决定是否补正则。
- `dedup`/`pred` 中 FAISS 替换为了 `sklearn.neighbors.NearestNeighbors`，保持跨平台轻量。

## 参考

- Niu et al., *Tree-KG: An Expandable Knowledge Graph Construction Framework for Knowledge-intensive Domains*, ACL 2025.
- 官方仓库：<https://github.com/thu-pacman/Tree-KG>
