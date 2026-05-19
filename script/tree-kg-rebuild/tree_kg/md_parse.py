r"""Markdown 输入解析。

很多扫描件 PDF 走 OCR 后会被转换成 Markdown，标题用 `#`/`##`/... 表示层级，
天然就是一棵树，不再需要复杂正则。

策略：
- 按行扫描，识别 `^(#{1,6})\s+(.+)$` 作为标题；其余行视为正文。
- 用栈维护当前路径：遇到新标题时弹出 level >= 新 level 的祖先，新节点挂到栈顶之下。
- 正文累积到"当前节点"。如果该节点之后又拥有子节点，把已有正文转为一个"引言"叶子（避免丢失），
  保持上层节点 raw_text 为空、由子节点摘要聚合（与 PDF 流程一致）。

输出与 pdf_parse.parse_pdf_to_tree 同构（TocNode 树），上层调度无差别。
"""

from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional

from .pdf_parse import TocNode


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _normalize(text: str) -> str:
    return text.strip().rstrip("#").strip()


def parse_markdown_text(
    md_text: str,
    *,
    material_name: str,
    max_level: int = 6,
    min_level: int = 1,
    skip_heading_patterns: Optional[List[str]] = None,
    strip_heading_pattern: Optional[str] = None,
    intro_leaf_suffix: str = " · 前言",
) -> TocNode:
    r"""把 Markdown 文本解析为 TocNode 树。

    Args:
        material_name:  作为根节点名（layer=book）
        max_level:      最深处理到几级标题（`#` 个数），超过当作正文
        min_level:      只接受 >= min_level 的标题作为节点；更高（即 `#` 更少，
                        level 更小）的标题忽略并视为正文。常用于丢掉 OCR 多余的封面标题。
        skip_heading_patterns:  匹配命中即跳过该标题及其整棵子树（用于"目录"等噪声段）
        strip_heading_pattern:  额外从标题文本剥离的正则（例如尾部页码 `\s+\d+$`）
        intro_leaf_suffix:      若内部节点直接有正文（且后续有子节点），会以
                                 "<name><intro_leaf_suffix>" 的名字另起一个叶子，避免丢内容
    """
    skip_res = [re.compile(p) for p in (skip_heading_patterns or [])]
    strip_re = re.compile(strip_heading_pattern) if strip_heading_pattern else None

    root = TocNode(name=material_name, level=0, children=[], raw_text="")
    stack: List[TocNode] = [root]
    current: TocNode = root
    pending_body: List[str] = []
    skip_until_level: Optional[int] = None  # 跳过子树到下一个 level <= 此值的标题为止

    def _flush_body(target: TocNode) -> None:
        if not pending_body:
            return
        text = "\n".join(pending_body).strip()
        pending_body.clear()
        if not text:
            return
        if target.raw_text:
            target.raw_text = target.raw_text + "\n" + text
        else:
            target.raw_text = text

    for raw_line in md_text.splitlines():
        m = _HEADING_RE.match(raw_line)
        if not m:
            if skip_until_level is None:
                pending_body.append(raw_line)
            continue

        level = len(m.group(1))
        name = _normalize(m.group(2))
        if strip_re:
            name = strip_re.sub("", name).strip()

        # 不在我们关心的层级范围内 → 当成普通文本
        if level > max_level or level < min_level:
            if skip_until_level is None:
                pending_body.append(raw_line)
            continue

        # 跳过子树
        if skip_until_level is not None:
            if level <= skip_until_level:
                skip_until_level = None
            else:
                continue

        if any(r.search(name) for r in skip_res):
            skip_until_level = level
            continue

        # 把累积的正文落地到当前节点
        _flush_body(current)

        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1]

        new_node = TocNode(name=name, level=level, children=[], raw_text="")
        parent.children.append(new_node)
        stack.append(new_node)
        current = new_node

    _flush_body(current)

    # 把"内部节点 + raw_text"分裂为引言叶子
    _split_intros(root, intro_leaf_suffix)
    return root


def parse_markdown_file(
    md_path: str,
    *,
    material_name: Optional[str] = None,
    max_level: int = 6,
    min_level: int = 1,
    skip_heading_patterns: Optional[List[str]] = None,
    strip_heading_pattern: Optional[str] = None,
) -> TocNode:
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    if material_name is None:
        material_name = os.path.splitext(os.path.basename(md_path))[0]
    return parse_markdown_text(
        text,
        material_name=material_name,
        max_level=max_level,
        min_level=min_level,
        skip_heading_patterns=skip_heading_patterns,
        strip_heading_pattern=strip_heading_pattern,
    )


def _split_intros(node: TocNode, suffix: str) -> None:
    """若 node 已有子节点又留了 raw_text，把 raw_text 提取为一个引言子叶子。"""
    if node.children and node.raw_text.strip():
        intro = TocNode(
            name=f"{node.name}{suffix}",
            level=node.level + 1,
            children=[],
            raw_text=node.raw_text,
        )
        node.raw_text = ""
        node.children.insert(0, intro)
    for c in list(node.children):
        _split_intros(c, suffix)
