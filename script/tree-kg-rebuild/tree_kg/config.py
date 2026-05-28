"""加载用户配置 (user_config.json) 和 API 配置 (api_config.json)。

api_config 设计为：
{
  "default":           {... 默认 chat 接入 ...},
  "default_embedding": {... 默认 embedding 接入 ...},
  "stages": {
      "summary": { "use": "default" }            -- 复用别名
      "extract": { "api_key": "...", ... }       -- 直接覆写
  },
  "runtime": {... 运行参数 ...}
}

支持的 stage 名:
- summary   bottom-up 摘要
- extract   实体/关系抽取
- conv      上下文卷积
- aggr      实体聚合 (core/non-core 判定)
- dedup     去重判同
- pred      边预测
- embed     向量化 (embedding 类型)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


VALID_CHAT_STAGES = ("summary", "extract", "conv", "aggr", "dedup", "pred")
VALID_EMBED_STAGES = ("embed",)
ALL_STAGES = VALID_CHAT_STAGES + VALID_EMBED_STAGES


@dataclass
class StageEndpoint:
    """单一阶段的 LLM/Embedding 接入信息。"""

    stage: str
    type: str  # "chat" or "embedding"
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: float = 120.0
    max_retries: int = 3
    retry_backoff: float = 2.0
    batch_size: int = 16
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserConfig:
    """统一的用户配置。兼容 PDF 与 Markdown 两类输入。

    - PDF 用 page 区间 + toc_re_expression 切分
    - Markdown 用 `#` 标题层级原生切分（无需正则）

    source_type 可由调用方显式覆盖（main.py 会按 --input/--markdown 自动判定）。
    """

    course_name: str
    material_name: str
    source_type: str = "pdf"  # "pdf" | "markdown"

    # ---------- PDF-specific（markdown 时全部忽略） ----------
    book_start_page: int = 1
    book_end_page: int = 0
    cover_start_page: int = 1
    cover_end_page: int = 1
    toc_start_page: int = 1
    toc_end_page: int = 1
    text_start_page: int = 1
    text_end_page: int = 0
    appendix_start_page: int = 0
    appendix_end_page: int = 0
    toc_max_level: int = 3
    toc_re_expression: List[str] = field(default_factory=list)

    # ---------- Markdown-specific（pdf 时忽略） ----------
    md_max_level: int = 6
    md_min_level: int = 1
    md_skip_heading_patterns: List[str] = field(default_factory=list)
    md_strip_heading_pattern: str = ""

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> "UserConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserConfig":
        source_type = str(data.get("source_type", "pdf")).lower()
        kwargs: Dict[str, Any] = {
            "source_type": source_type,
            "raw": data,
        }

        if source_type == "pdf":
            # PDF 模式下 course/material 名称必填，因为页码/正则也是用户必须显式指定的
            kwargs["course_name"] = data["course_name"]
            kwargs["material_name"] = data["material_name"]
            for f_name in (
                "book_start_page", "book_end_page",
                "cover_start_page", "cover_end_page",
                "toc_start_page", "toc_end_page",
                "text_start_page", "text_end_page",
                "appendix_start_page", "appendix_end_page",
                "toc_max_level",
            ):
                if f_name in data:
                    kwargs[f_name] = int(data[f_name])
            if "toc_re_expression" in data:
                kwargs["toc_re_expression"] = list(data["toc_re_expression"])

        elif source_type == "markdown":
            # Markdown 模式下 course/material 名称可选，缺省由 main.py 用文件名补齐
            if "course_name" in data:
                kwargs["course_name"] = data["course_name"]
            if "material_name" in data:
                kwargs["material_name"] = data["material_name"]
            if "md_max_level" in data:
                kwargs["md_max_level"] = int(data["md_max_level"])
            if "md_min_level" in data:
                kwargs["md_min_level"] = int(data["md_min_level"])
            if "md_skip_heading_patterns" in data:
                kwargs["md_skip_heading_patterns"] = list(data["md_skip_heading_patterns"])
            if "md_strip_heading_pattern" in data:
                kwargs["md_strip_heading_pattern"] = str(data["md_strip_heading_pattern"])
        else:
            raise ValueError(f"未知的 source_type: {source_type!r}（仅支持 pdf / markdown）")

        # 兜底
        kwargs.setdefault("course_name", "")
        kwargs.setdefault("material_name", "")
        return cls(**kwargs)

    @classmethod
    def default_for_markdown(cls, source_path: str) -> "UserConfig":
        """Markdown 模式下的零配置默认值：所有名字都从文件名取。"""
        stem = os.path.splitext(os.path.basename(source_path))[0]
        return cls(
            course_name=stem,
            material_name=stem,
            source_type="markdown",
            raw={"source_type": "markdown", "_synthetic": True, "_from": source_path},
        )


@dataclass
class RuntimeConfig:
    concurrency: int = 5
    checkpoint_dir: str = "./output/checkpoints"
    output_dir: str = "./output"
    dedup_threshold: float = 0.55
    edge_pred_strength_threshold: int = 6
    edge_pred_extra_edges: int = 1000
    edge_pred_alpha: float = 0.6
    edge_pred_beta_stage1: float = 0.0
    edge_pred_gamma_stage1: float = 0.4
    edge_pred_beta_stage2: float = 0.3
    edge_pred_gamma_stage2: float = 0.1


@dataclass
class ApiConfig:
    stages: Dict[str, StageEndpoint]
    runtime: RuntimeConfig
    raw: Dict[str, Any] = field(default_factory=dict)

    def endpoint(self, stage: str) -> StageEndpoint:
        if stage not in self.stages:
            raise KeyError(
                f"未为阶段 '{stage}' 配置 API。请在 api_config.json 的 stages 中添加。"
            )
        return self.stages[stage]

    @classmethod
    def from_file(cls, path: str) -> "ApiConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApiConfig":
        runtime_raw = data.get("runtime", {}) or {}
        runtime = RuntimeConfig(**{k: v for k, v in runtime_raw.items() if k in RuntimeConfig.__dataclass_fields__})

        aliases: Dict[str, Dict[str, Any]] = {}
        for key, val in data.items():
            if key in ("stages", "runtime"):
                continue
            if isinstance(val, dict) and not key.startswith("_"):
                aliases[key] = val

        stages_def: Dict[str, Dict[str, Any]] = data.get("stages", {}) or {}
        if not stages_def:
            stages_def = {s: {"use": "default"} for s in VALID_CHAT_STAGES}
            stages_def["embed"] = {"use": "default_embedding"}

        endpoints: Dict[str, StageEndpoint] = {}
        for stage in ALL_STAGES:
            cfg = dict(stages_def.get(stage, {}))
            if "use" in cfg:
                base = aliases.get(cfg.pop("use"))
                if base is None:
                    raise ValueError(
                        f"阶段 {stage} 引用了未定义的别名: {cfg.get('use')}"
                    )
                merged = dict(base)
                merged.update(cfg)
                cfg = merged
            elif not cfg:
                fallback = (
                    aliases.get("default_embedding") if stage in VALID_EMBED_STAGES else aliases.get("default")
                )
                if fallback is None:
                    raise ValueError(f"阶段 {stage} 未配置接入信息且无默认值。")
                cfg = dict(fallback)
            endpoints[stage] = _make_endpoint(stage, cfg)

        return cls(stages=endpoints, runtime=runtime, raw=data)


def _make_endpoint(stage: str, cfg: Dict[str, Any]) -> StageEndpoint:
    api_key = cfg.get("api_key") or os.environ.get(cfg.get("api_key_env", ""), "")
    if not api_key or api_key == "REPLACE_ME":
        raise ValueError(
            f"阶段 {stage} 缺少有效 api_key。请在 api_config.json 中配置或通过 api_key_env 引用环境变量。"
        )
    expected_type = "embedding" if stage in VALID_EMBED_STAGES else "chat"
    return StageEndpoint(
        stage=stage,
        type=cfg.get("type", expected_type),
        api_key=api_key,
        base_url=cfg.get("base_url", "https://api.openai.com/v1"),
        model=cfg["model"],
        temperature=float(cfg.get("temperature", 0.2)),
        max_tokens=int(cfg.get("max_tokens", 4096)),
        timeout=float(cfg.get("timeout", 120)),
        max_retries=int(cfg.get("max_retries", 3)),
        retry_backoff=float(cfg.get("retry_backoff", 2.0)),
        batch_size=int(cfg.get("batch_size", 16)),
        extra={k: v for k, v in cfg.items() if k not in {
            "type", "api_key", "api_key_env", "base_url", "model", "temperature",
            "max_tokens", "timeout", "max_retries", "retry_backoff", "batch_size",
        }},
    )


def load_configs(user_path: str, api_path: str) -> tuple[UserConfig, ApiConfig]:
    return UserConfig.from_file(user_path), ApiConfig.from_file(api_path)
