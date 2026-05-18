"""PDF 文本提取 + 基于正则的 TOC 切分。

页码协议：用户配置中的页码均为 PDF 文件页（1-based），与书内编号无关，与 GitHub 文档一致。

切分策略：
- 提取 [text_start_page, text_end_page] 范围内的文本
- 按用户给的 toc_re_expression 列表，level 1 -> level k 顺次递归切分
- 每一级的切分点把文本切分为若干 (heading, body) 段
- 把每个段 body 再用下一级正则切分；若没有更深层级或下一级无匹配，则该段为叶子节点（保留 raw_text）

返回：层级化的 dict 树，便于上层映射到 KnowledgeGraph。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise ImportError("缺少 PyMuPDF。请执行 pip install -r requirements.txt") from e


@dataclass
class TocNode:
    """TOC 切分得到的层级树节点。"""
    name: str
    level: int               # 0 = book 根
    children: List["TocNode"] = field(default_factory=list)
    raw_text: str = ""       # 仅叶子上有正文

    def is_leaf(self) -> bool:
        return not self.children

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


def extract_text_by_pages(pdf_path: str, start: int, end: int) -> str:
    """提取 [start, end] 闭区间页（1-based）的纯文本，每页之间双换行。"""
    doc = fitz.open(pdf_path)
    try:
        pages = []
        last = min(end, doc.page_count)
        for i in range(max(1, start) - 1, last):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            pages.append(text)
        return "\n\n".join(pages)
    finally:
        doc.close()


def _split_by_regex(text: str, pattern: str) -> List[Dict[str, str]]:
    """按 *捕获组* 正则切分文本，返回 [{"heading": ..., "body": ...}, ...]。

    规则：
    - 以 finditer 找到所有 heading；前面那段属于"上一段"。
    - heading 前如果还有正文（即第一个 heading 之前），作为 prelude 单独丢弃。
    """
    rgx = re.compile(pattern, flags=re.MULTILINE)
    matches = list(rgx.finditer(text))
    if not matches:
        return []
    sections: List[Dict[str, str]] = []
    for i, m in enumerate(matches):
        head = m.group(1) if m.groups() else m.group(0)
        head = head.strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        if not head:
            continue
        sections.append({"heading": head, "body": body})
    return sections


def _build_tree(text: str, regexes: List[str], level: int) -> List[TocNode]:
    if level >= len(regexes) or not text.strip():
        return []
    pattern = regexes[level]
    parts = _split_by_regex(text, pattern)
    if not parts:
        return []
    out: List[TocNode] = []
    for p in parts:
        node = TocNode(name=p["heading"], level=level + 1)
        sub_children = _build_tree(p["body"], regexes, level + 1)
        if sub_children:
            node.children = sub_children
            # 叶子的正文交给最深一级；中间层不保留 raw
        else:
            node.raw_text = p["body"].strip()
        out.append(node)
    return out


def parse_pdf_to_tree(pdf_path: str, *, course: str, material: str,
                       text_start_page: int, text_end_page: int,
                       toc_re_expression: List[str], toc_max_level: int) -> TocNode:
    """主入口：返回根节点 TocNode（根名 = 教材名）。"""
    body = extract_text_by_pages(pdf_path, text_start_page, text_end_page)
    regexes = toc_re_expression[:toc_max_level]
    children = _build_tree(body, regexes, level=0)
    root = TocNode(name=material or course, level=0, children=children)
    return root


def collect_toc_text(pdf_path: str, toc_start_page: int, toc_end_page: int) -> str:
    """提取 toc 页文本（仅供调试或可视化）。"""
    return extract_text_by_pages(pdf_path, toc_start_page, toc_end_page)


# --------------------------------------------------------------- diagnostics

def tree_summary(tree: TocNode) -> Dict[str, Any]:
    counts: Dict[int, int] = {}
    leaves = 0
    raw_chars = 0
    for n in tree.walk():
        counts[n.level] = counts.get(n.level, 0) + 1
        if n.is_leaf():
            leaves += 1
            raw_chars += len(n.raw_text)
    return {
        "level_counts": counts,
        "leaves": leaves,
        "total_raw_chars": raw_chars,
    }


def render_outline(tree: TocNode, max_lines: int = 80) -> str:
    """输出大纲预览，便于人工核对正则匹配是否合理。"""
    lines: List[str] = []
    def _walk(n: TocNode, depth: int) -> None:
        if len(lines) >= max_lines:
            return
        prefix = "  " * depth + "- "
        lines.append(prefix + n.name[:80])
        for c in n.children:
            _walk(c, depth + 1)
    _walk(tree, 0)
    if len(lines) >= max_lines:
        lines.append("...")
    return "\n".join(lines)
