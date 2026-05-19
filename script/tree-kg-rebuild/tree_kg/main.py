"""命令行入口。

示例：
    # PDF 输入
    python -m tree_kg \
        --input path/to/book.pdf \
        --user-config configs/user_config.json \
        --api-config  configs/api_config.json

    # Markdown 输入（如：OCR 后得到的 .md）
    python -m tree_kg \
        --input path/to/book.md \
        --user-config configs/user_config.markdown.json \
        --api-config  configs/api_config.json

可选：
    --pdf / --markdown   显式指定输入类型（也可由扩展名/配置文件自动判断）
    --skip-phase2        仅运行 Phase 1
    --resume PATH        从已有 KG checkpoint 继续 (跳过 Phase 1)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from . import logger
from .config import ApiConfig, UserConfig
from .graph import KnowledgeGraph
from .phase1 import run_phase1
from .phase2 import run_phase2


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _detect_source_type(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext in (".md", ".markdown", ".mkd"):
        return "markdown"
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tree_kg",
        description="Tree-KG: 知识图谱构建框架（论文复现，支持 PDF / Markdown 输入）",
    )
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument("--input", "-i", default=None, help="教材源文件路径（按扩展名自动识别 pdf / markdown）")
    src_group.add_argument("--pdf", default=None, help="教材 PDF 路径（等价于 --input xxx.pdf）")
    src_group.add_argument("--markdown", "--md", dest="markdown", default=None,
                           help="教材 Markdown 路径（OCR 转换得到的 .md 也适用）")

    parser.add_argument("--user-config", default=None,
                        help="用户配置 JSON（Markdown 模式下可省略，会用文件名作为默认 course/material）")
    parser.add_argument("--api-config", required=True, help="LLM 接入配置 JSON")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认读取 api_config.runtime.output_dir）")
    parser.add_argument("--skip-phase2", action="store_true", help="只跑 Phase 1（初始构建）")
    parser.add_argument("--resume", default=None, help="从已有 KG json 文件加载并直接进入 Phase 2")

    args = parser.parse_args(argv)

    forced_type: Optional[str] = None
    source_path: Optional[str] = None
    if args.pdf:
        source_path = args.pdf
        forced_type = "pdf"
    elif args.markdown:
        source_path = args.markdown
        forced_type = "markdown"
    elif args.input:
        source_path = args.input

    if not args.resume:
        if not source_path:
            logger.error("必须通过 --input / --pdf / --markdown 指定源文件。")
            return 2
        if not os.path.isfile(source_path):
            logger.error(f"源文件不存在: {source_path}")
            return 2
    if not os.path.isfile(args.api_config):
        logger.error(f"API 配置不存在: {args.api_config}")
        return 2

    detected_type = _detect_source_type(source_path) if source_path else None
    effective_type = forced_type or detected_type

    if args.user_config:
        if not os.path.isfile(args.user_config):
            logger.error(f"用户配置不存在: {args.user_config}")
            return 2
        user_cfg = UserConfig.from_file(args.user_config)
    else:
        if effective_type != "markdown":
            logger.error("--user-config 仅在 Markdown 模式下可省略。PDF 模式必须提供用户配置。")
            return 2
        logger.info("未提供 --user-config，Markdown 模式下使用文件名作为默认 course/material。")
        user_cfg = UserConfig.default_for_markdown(source_path)

    if forced_type:
        if user_cfg.source_type != forced_type:
            logger.warn(
                f"用户配置 source_type={user_cfg.source_type!r}，被 CLI 覆盖为 {forced_type!r}"
            )
            user_cfg.source_type = forced_type
    elif source_path:
        if detected_type and detected_type != user_cfg.source_type:
            logger.warn(
                f"扩展名暗示 source_type={detected_type!r}，与配置 {user_cfg.source_type!r} 不一致，按扩展名采用。"
            )
            user_cfg.source_type = detected_type

    # markdown 模式下若未填 course/material，自动用文件名补齐
    if user_cfg.source_type == "markdown" and source_path:
        stem = os.path.splitext(os.path.basename(source_path))[0]
        if not user_cfg.course_name:
            user_cfg.course_name = stem
        if not user_cfg.material_name:
            user_cfg.material_name = stem

    api_cfg = ApiConfig.from_file(args.api_config)
    out_dir = _ensure_dir(args.output_dir or api_cfg.runtime.output_dir)
    ckpt_dir = _ensure_dir(api_cfg.runtime.checkpoint_dir)

    logger.section("Tree-KG 启动")
    logger.info(f"course={user_cfg.course_name} material={user_cfg.material_name}")
    logger.info(f"source_type={user_cfg.source_type} source={source_path}")
    logger.info(f"output_dir={out_dir}")
    logger.info(f"checkpoint_dir={ckpt_dir}")
    started = time.time()

    if args.resume:
        if not os.path.isfile(args.resume):
            logger.error(f"resume 文件不存在: {args.resume}")
            return 2
        logger.step(f"恢复 KG: {args.resume}")
        kg = KnowledgeGraph.load(args.resume)
        logger.info(f"加载完毕: {kg.stats()}")
    else:
        try:
            kg = run_phase1(source_path, user_cfg, api_cfg, checkpoint_path=ckpt_dir)
        except KeyboardInterrupt:
            logger.error("用户中断 Phase 1。")
            return 130

    if not args.skip_phase2:
        try:
            kg = run_phase2(kg, api_cfg, user_cfg.course_name, checkpoint_path=ckpt_dir)
        except KeyboardInterrupt:
            logger.error("用户中断 Phase 2，已尝试在 checkpoint 目录保留中间结果。")
            return 130
    else:
        logger.warn("已 --skip-phase2，跳过迭代扩展。")

    final_path = os.path.join(out_dir, f"kg_{user_cfg.course_name}.json")
    kg.save(final_path)
    elapsed = time.time() - started
    logger.section("完成")
    logger.ok(f"最终 KG 已保存：{final_path}")
    logger.ok(f"统计：{kg.stats()}")
    logger.ok(f"总耗时：{elapsed:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
