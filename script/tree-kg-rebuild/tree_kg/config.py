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
    course_name: str
    material_name: str
    book_start_page: int
    book_end_page: int
    cover_start_page: int
    cover_end_page: int
    toc_start_page: int
    toc_end_page: int
    text_start_page: int
    text_end_page: int
    appendix_start_page: int
    appendix_end_page: int
    toc_max_level: int
    toc_re_expression: List[str]
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> "UserConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            course_name=data["course_name"],
            material_name=data["material_name"],
            book_start_page=int(data["book_start_page"]),
            book_end_page=int(data["book_end_page"]),
            cover_start_page=int(data["cover_start_page"]),
            cover_end_page=int(data["cover_end_page"]),
            toc_start_page=int(data["toc_start_page"]),
            toc_end_page=int(data["toc_end_page"]),
            text_start_page=int(data["text_start_page"]),
            text_end_page=int(data["text_end_page"]),
            appendix_start_page=int(data["appendix_start_page"]),
            appendix_end_page=int(data["appendix_end_page"]),
            toc_max_level=int(data["toc_max_level"]),
            toc_re_expression=list(data["toc_re_expression"]),
            raw=data,
        )


@dataclass
class RuntimeConfig:
    concurrency: int = 4
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
