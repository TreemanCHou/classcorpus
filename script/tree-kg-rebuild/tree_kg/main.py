"""命令行入口。

示例：
    python -m tree_kg.main \
        --pdf path/to/book.pdf \
        --user-config configs/user_config.json \
        --api-config configs/api_config.json \
        --output-dir ./output

可选：
    --skip-phase2     仅运行 Phase 1
    --resume PATH     从已有 KG checkpoint 继续 (跳过 Phase 1)
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tree_kg", description="Tree-KG: 知识图谱构建框架（论文复现）")
    parser.add_argument("--pdf", required=True, help="教材 PDF 路径")
    parser.add_argument("--user-config", required=True, help="用户配置 JSON")
    parser.add_argument("--api-config", required=True, help="LLM 接入配置 JSON")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认读取 api_config.runtime.output_dir）")
    parser.add_argument("--skip-phase2", action="store_true", help="只跑 Phase 1（初始构建）")
    parser.add_argument("--resume", default=None, help="从已有 KG json 文件加载并直接进入 Phase 2")

    args = parser.parse_args(argv)

    if not os.path.isfile(args.pdf) and not args.resume:
        logger.error(f"PDF 不存在: {args.pdf}")
        return 2
    if not os.path.isfile(args.user_config):
        logger.error(f"用户配置不存在: {args.user_config}")
        return 2
    if not os.path.isfile(args.api_config):
        logger.error(f"API 配置不存在: {args.api_config}")
        return 2

    user_cfg = UserConfig.from_file(args.user_config)
    api_cfg = ApiConfig.from_file(args.api_config)
    out_dir = _ensure_dir(args.output_dir or api_cfg.runtime.output_dir)
    ckpt_dir = _ensure_dir(api_cfg.runtime.checkpoint_dir)

    logger.section("Tree-KG 启动")
    logger.info(f"course={user_cfg.course_name} material={user_cfg.material_name}")
    logger.info(f"PDF={args.pdf}")
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
            kg = run_phase1(args.pdf, user_cfg, api_cfg, checkpoint_path=ckpt_dir)
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
