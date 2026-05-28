"""LLM 调用统一封装。

要求：
1. 不让上层直接 import openai；后续如果要换 SDK，只需改本模块。
2. 同时提供 chat (JSON 模式) 与 embedding 两类接口。
3. 错误分类：
   - **可恢复网络错误**（连接断开、超时、5xx、Rate limit）→ 自动重试。
   - **资源型错误**（额度不足、auth 失败、欠费 402、403）→ 给出提示并自动重试。
   - **多次失败** → 抛给阶段逻辑，由阶段逻辑决定重试当前项或跳过。
4. 提供 stage 维度路由：根据 ApiConfig 选择对应 endpoint。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import logger
from .config import ApiConfig, StageEndpoint

try:
    from openai import OpenAI
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        InternalServerError,
        PermissionDeniedError,
        RateLimitError,
    )
    try:
        from openai import APIStatusError  # 新版 SDK
    except ImportError:  # 老版本兼容
        APIStatusError = Exception  # type: ignore[assignment]
except ImportError as e:
    raise ImportError(
        "未找到 openai 库。请先运行: pip install -r requirements.txt"
    ) from e


_RETRYABLE_EXC = (APIConnectionError, APITimeoutError, InternalServerError)
_PAUSABLE_EXC = (AuthenticationError, PermissionDeniedError, RateLimitError)


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMClient:
    """按 stage 路由的 LLM/Embedding 调用入口。

    使用：
        client = LLMClient(api_cfg)
        text = client.chat("summary", messages=[...])
        vec = client.embed("embed", inputs=[...])
    """

    def __init__(self, api_cfg: ApiConfig):
        self.api_cfg = api_cfg
        self._clients: Dict[str, OpenAI] = {}

    def endpoint(self, stage: str) -> StageEndpoint:
        return self.api_cfg.endpoint(stage)

    def _client_for(self, ep: StageEndpoint) -> OpenAI:
        cache_key = f"{ep.base_url}|{ep.api_key[:8]}"
        cli = self._clients.get(cache_key)
        if cli is None:
            cli = OpenAI(
                api_key=ep.api_key,
                base_url=ep.base_url,
                timeout=ep.timeout,
                max_retries=0,  # 我们自行控制重试
            )
            self._clients[cache_key] = cli
        return cli

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        stage: str,
        messages: List[ChatMessage] | List[Dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        ep = self.endpoint(stage)
        if ep.type != "chat":
            raise ValueError(f"阶段 {stage} 的 endpoint 类型不是 chat ({ep.type})。")

        msgs = [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages]

        kwargs: Dict[str, Any] = {
            "model": ep.model,
            "messages": msgs,
            "temperature": ep.temperature if temperature is None else temperature,
            "max_tokens": ep.max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        return self._invoke_with_recovery(
            stage=stage,
            ep=ep,
            label="chat",
            fn=lambda: self._client_for(ep).chat.completions.create(**kwargs).choices[0].message.content or "",
        )

    def chat_json(
        self,
        stage: str,
        messages: List[ChatMessage] | List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        try_native_json: bool = True,
    ) -> Any:
        """要求模型返回 JSON 并解析。

        - 优先使用 response_format=json_object（若服务端支持）。
        - 失败 (BadRequestError) 则回退到 plain，再用正则提取。
        """
        attempts = 0
        while True:
            attempts += 1
            try:
                try:
                    if try_native_json:
                        raw = self.chat(
                            stage,
                            messages,
                            json_mode=True,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    else:
                        raw = self.chat(stage, messages, json_mode=False, temperature=temperature, max_tokens=max_tokens)
                except BadRequestError as e:
                    logger.warn(f"[{stage}] 服务端不支持 json_object 模式，降级为纯文本: {e}")
                    raw = self.chat(stage, messages, json_mode=False, temperature=temperature, max_tokens=max_tokens)
                return _extract_json(raw)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                if attempts < 3:
                    logger.warn(
                        f"[{stage}/json] JSON 解析/调用失败 ({type(e).__name__}): {e} | "
                        f"2.0s 后重试 ({attempts}/3)"
                    )
                    time.sleep(2.0)
                    continue
                raise

    # ------------------------------------------------------------------- embed
    def embed(self, stage: str, inputs: List[str]) -> List[List[float]]:
        ep = self.endpoint(stage)
        if ep.type != "embedding":
            raise ValueError(f"阶段 {stage} 的 endpoint 类型不是 embedding ({ep.type})。")
        if not inputs:
            return []

        out: List[List[float]] = []
        bs = max(1, ep.batch_size)
        for i in range(0, len(inputs), bs):
            chunk = inputs[i: i + bs]
            res = self._invoke_with_recovery(
                stage=stage,
                ep=ep,
                label=f"embed[{i}:{i + len(chunk)}]",
                fn=lambda c=chunk: self._client_for(ep).embeddings.create(model=ep.model, input=c),
            )
            out.extend([d.embedding for d in res.data])
        return out

    # --------------------------------------------------------- recovery loop
    def _invoke_with_recovery(self, *, stage: str, ep: StageEndpoint, label: str, fn):
        """统一调用包装：失败时自动重试 3 次，仍失败则向上抛出。"""
        attempts = 0
        while True:
            attempts += 1
            try:
                return fn()
            except _RETRYABLE_EXC as e:
                if attempts < 3:
                    logger.warn(
                        f"[{stage}/{label}] 网络/服务波动 ({type(e).__name__}): {e} | "
                        f"2.0s 后重试 ({attempts}/3)"
                    )
                    time.sleep(2.0)
                    continue
                raise
            except _PAUSABLE_EXC as e:
                msg = str(e)
                hint = _resource_hint(e, msg)
                logger.error(f"[{stage}/{label}] {type(e).__name__}: {msg}")
                logger.warn(f"提示：{hint}")
                if attempts < 3:
                    logger.warn(f"[{stage}/{label}] 2.0s 后重试 ({attempts}/3)")
                    time.sleep(2.0)
                    continue
                raise
            except APIStatusError as e:  # type: ignore[misc]
                status = getattr(e, "status_code", None)
                if status in (401, 402, 403, 429):
                    logger.error(f"[{stage}/{label}] HTTP {status}: {e}")
                    logger.warn("可能是鉴权失败/欠费/限频，请处理后继续。")
                    if attempts < 3:
                        logger.warn(f"[{stage}/{label}] 2.0s 后重试 ({attempts}/3)")
                        time.sleep(2.0)
                        continue
                    raise
                if status and 500 <= status < 600 and attempts < 3:
                    logger.warn(f"[{stage}/{label}] 服务端 5xx: {e}，2.0s 后重试 ({attempts}/3)")
                    time.sleep(2.0)
                    continue
                raise
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error(f"[{stage}/{label}] 未预期错误 {type(e).__name__}: {e}")
                if attempts < 3:
                    logger.warn(f"[{stage}/{label}] 2.0s 后重试 ({attempts}/3)")
                    time.sleep(2.0)
                    continue
                raise


def _resource_hint(exc: Exception, msg: str) -> str:
    lower = msg.lower()
    if isinstance(exc, AuthenticationError) or "auth" in lower or "401" in lower:
        return "鉴权失败：请检查 api_key 是否正确、是否过期。"
    if isinstance(exc, PermissionDeniedError) or "402" in lower or "balance" in lower or "余额" in msg:
        return "余额不足或权限不足：请充值 / 更换可用 API Key。"
    if isinstance(exc, RateLimitError) or "429" in lower or "rate" in lower:
        return "触发限频：请等待一段时间或更换 API Key。"
    return "请检查 API 服务可用性、Key、配额后再继续。"


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> Any:
    """从模型输出中宽松提取 JSON。

    优先级：
    1) 直接 json.loads
    2) 去 markdown 代码块再 loads
    3) 从首个 `{` 或 `[` 截取到最后一个 `}` 或 `]`
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("LLM 返回为空。")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(text)
    if m:
        snippet = m.group(1).strip()
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i = text.find(opener)
        j = text.rfind(closer)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i: j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"无法从 LLM 响应解析 JSON: {text[:200]} ...")
