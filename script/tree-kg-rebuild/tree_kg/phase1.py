"""Phase 1: 初始构建。

步骤：
1. PDF -> TOC 树
2. TOC 树 -> KG 节点 + has_subsection 垂直边
3. 自下而上摘要：叶子节点用 LLM 生成摘要；上层节点用子节点摘要聚合
4. 每个叶子节点：抽取实体（has_entity 边）+ 实体内部关系（entity_related 水平边）
5. 每个非叶 TOC 节点：抽取子章节关系（section_related 水平边）
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from . import logger, prompts
from .config import ApiConfig, UserConfig
from .graph import (
    Edge,
    HORIZ_ENTITY_RELATED,
    HORIZ_SECTION_RELATED,
    KnowledgeGraph,
    LAYER_BOOK,
    LAYER_CHAPTER_OR_SECTION,
    LAYER_ENTITY_CORE,
    LAYER_SUBSECTION_LEAF,
    Node,
    VERT_HAS_ENTITY,
    VERT_HAS_SUBSECTION,
)
from .llm import LLMClient
from .md_parse import parse_markdown_file
from .pdf_parse import TocNode, parse_pdf_to_tree, render_outline, tree_summary


def _toc_layer(level: int, is_leaf: bool) -> str:
    if level == 0:
        return LAYER_BOOK
    if is_leaf:
        return LAYER_SUBSECTION_LEAF
    return LAYER_CHAPTER_OR_SECTION


def build_kg_skeleton(tree: TocNode) -> KnowledgeGraph:
    """把 TOC 树转换为 KG 节点 + has_subsection 边。"""
    kg = KnowledgeGraph()

    def _emit(node: TocNode, parent_id: Optional[str]) -> str:
        layer = _toc_layer(node.level, node.is_leaf())
        nid = kg.new_id(prefix={
            LAYER_BOOK: "book",
            LAYER_CHAPTER_OR_SECTION: "sec",
            LAYER_SUBSECTION_LEAF: "leaf",
        }.get(layer, "n"))
        kg.add_node(Node(
            id=nid,
            name=node.name,
            layer=layer,
            depth=node.level,
            raw_text=node.raw_text if node.is_leaf() else "",
        ))
        if parent_id is not None:
            kg.add_edge(Edge(
                src=parent_id,
                dst=nid,
                kind="vertical",
                category=VERT_HAS_SUBSECTION,
            ))
        for c in node.children:
            _emit(c, nid)
        return nid

    _emit(tree, None)
    return kg


def section_path(kg: KnowledgeGraph, node_id: str) -> str:
    lineage = kg.lineage(node_id)
    return " / ".join(kg.nodes[i].name for i in reversed(lineage))


# ------------------------------------------------------------ summarization

def bottom_up_summarize(kg: KnowledgeGraph, llm: LLMClient, course: str) -> None:
    """自下而上，给所有 TOC 节点写 description（即论文中的 summary）。"""
    toc_layers = {LAYER_BOOK, LAYER_CHAPTER_OR_SECTION, LAYER_SUBSECTION_LEAF}
    nodes = [n for n in kg.nodes.values() if n.layer in toc_layers]
    nodes.sort(key=lambda n: -n.depth)

    leaves = [n for n in nodes if n.layer == LAYER_SUBSECTION_LEAF]
    inners = [n for n in nodes if n.layer != LAYER_SUBSECTION_LEAF]

    logger.subsection("Phase1.A · 自下而上摘要 - 叶子节点")
    for leaf in tqdm(leaves, desc="leaf summary", unit="sec"):
        if leaf.description:
            continue
        if not leaf.raw_text.strip():
            leaf.description = ""
            continue
        parent_id = kg.parent(leaf.id)
        parent_summary = kg.nodes[parent_id].description if parent_id else ""
        try:
            res = llm.chat_json("summary", prompts.summary_messages(course, parent_summary, leaf.raw_text))
            leaf.description = (res.get("summary") if isinstance(res, dict) else str(res)) or ""
        except Exception as e:
            logger.warn(f"叶子摘要失败 {leaf.name[:30]}: {e}")
            logger.pause("是否跳过？回车继续，下一个叶子；或 Ctrl+C 终止。")
            leaf.description = ""

    logger.subsection("Phase1.B · 自下而上摘要 - 上层聚合")
    for n in tqdm(inners, desc="inner summary", unit="sec"):
        children_ids = [c for c in kg.children(n.id) if kg.nodes[c].layer in toc_layers]
        child_sums = [kg.nodes[c].description for c in children_ids if kg.nodes[c].description]
        if not child_sums:
            n.description = ""
            continue
        try:
            res = llm.chat_json("summary", prompts.aggregate_summary_messages(course, child_sums))
            n.description = (res.get("summary") if isinstance(res, dict) else str(res)) or ""
        except Exception as e:
            logger.warn(f"父节点摘要失败 {n.name[:30]}: {e}")
            logger.pause("回车继续。")
            n.description = ""


# ----------------------------------------------------------- entity / relation

def extract_entities_and_relations(kg: KnowledgeGraph, llm: LLMClient, course: str) -> None:
    """对每个叶子 TOC 节点，抽取实体 + has_entity 垂直边 + entity_related 水平边。"""
    leaves = kg.nodes_at_layer(LAYER_SUBSECTION_LEAF)
    logger.subsection(f"Phase1.C · 实体/关系抽取（{len(leaves)} 个叶子节点）")

    for leaf in tqdm(leaves, desc="entity/relation", unit="sec"):
        if not leaf.description:
            continue
        path = section_path(kg, leaf.id)
        # 1) entities
        try:
            ent_res = llm.chat_json(
                "extract", prompts.entity_extract_messages(course, path, leaf.description)
            )
            ent_list = (ent_res or {}).get("entities", []) if isinstance(ent_res, dict) else []
        except Exception as e:
            logger.warn(f"实体抽取失败 {leaf.name[:30]}: {e}")
            logger.pause("回车继续。")
            continue

        # 节点：实体先以 core 默认入图（aggr 阶段会改 role）
        leaf_entity_ids: List[Tuple[str, Dict[str, Any]]] = []
        for ent in ent_list:
            name = (ent.get("name") or "").strip()
            if not name:
                continue
            nid = kg.new_id(prefix="ent")
            node = Node(
                id=nid,
                name=name,
                layer=LAYER_ENTITY_CORE,
                depth=leaf.depth + 1,
                description=ent.get("raw_content") or "",
                aliases=list(ent.get("alias") or []),
                role="core",
                meta={"type": ent.get("type", ""), "source_section": leaf.id},
            )
            kg.add_node(node)
            kg.add_edge(Edge(
                src=leaf.id,
                dst=nid,
                kind="vertical",
                category=VERT_HAS_ENTITY,
            ))
            leaf_entity_ids.append((nid, ent))

        if len(leaf_entity_ids) < 2:
            continue

        # 2) relations
        try:
            rel_res = llm.chat_json(
                "extract",
                prompts.relation_extract_messages(
                    course, path, leaf.description,
                    [e for _, e in leaf_entity_ids],
                ),
            )
            rels = (rel_res or {}).get("relations", []) if isinstance(rel_res, dict) else []
        except Exception as e:
            logger.warn(f"关系抽取失败 {leaf.name[:30]}: {e}")
            logger.pause("回车继续。")
            rels = []

        name_to_id = {ent["name"]: nid for nid, ent in leaf_entity_ids}
        for r in rels:
            sub = r.get("subject")
            obj = r.get("object")
            if not sub or not obj or sub == obj:
                continue
            sid = name_to_id.get(sub)
            did = name_to_id.get(obj)
            if not sid or not did:
                continue
            try:
                strength = float(r.get("strength", 0))
            except (TypeError, ValueError):
                strength = 0.0
            kg.add_edge(Edge(
                src=sid,
                dst=did,
                kind="horizontal",
                category=HORIZ_ENTITY_RELATED,
                relation=str(r.get("type", "")),
                description=str(r.get("description", "")),
                strength=strength,
            ))


def extract_section_relations(kg: KnowledgeGraph, llm: LLMClient, course: str) -> None:
    """对每个有多个 TOC 子节点的节点，抽取子节点之间的 section_related 关系。"""
    toc_layers = {LAYER_BOOK, LAYER_CHAPTER_OR_SECTION, LAYER_SUBSECTION_LEAF}
    inners = [n for n in kg.nodes.values() if n.layer in {LAYER_BOOK, LAYER_CHAPTER_OR_SECTION}]
    logger.subsection(f"Phase1.D · 章节关系抽取（{len(inners)} 个父节点）")

    for n in tqdm(inners, desc="section relations", unit="sec"):
        children_ids = [c for c in kg.children(n.id) if kg.nodes[c].layer in toc_layers]
        if len(children_ids) < 2:
            continue
        cdata = [
            {"name": kg.nodes[c].name, "summary": kg.nodes[c].description or ""}
            for c in children_ids
        ]
        path = section_path(kg, n.id)
        try:
            res = llm.chat_json(
                "extract",
                prompts.section_relation_messages(course, path, n.description or "", cdata),
            )
            rels = (res or {}).get("relations", []) if isinstance(res, dict) else []
        except Exception as e:
            logger.warn(f"章节关系抽取失败 {n.name[:30]}: {e}")
            logger.pause("回车继续。")
            continue

        name_to_id = {kg.nodes[c].name: c for c in children_ids}
        for r in rels:
            sub = r.get("subject")
            obj = r.get("object")
            if not sub or not obj or sub == obj:
                continue
            sid = name_to_id.get(sub)
            did = name_to_id.get(obj)
            if not sid or not did:
                continue
            try:
                strength = float(r.get("strength", 0))
            except (TypeError, ValueError):
                strength = 0.0
            kg.add_edge(Edge(
                src=sid,
                dst=did,
                kind="horizontal",
                category=HORIZ_SECTION_RELATED,
                relation=str(r.get("type", "")),
                description=str(r.get("description", "")),
                strength=strength,
            ))


# ------------------------------------------------------------------- driver

def parse_source_to_tree(source_path: str, user_cfg: UserConfig) -> TocNode:
    """根据 user_cfg.source_type 调度对应解析器，统一返回 TocNode。"""
    if user_cfg.source_type == "markdown":
        return parse_markdown_file(
            source_path,
            material_name=user_cfg.material_name or None,
            max_level=user_cfg.md_max_level,
            min_level=user_cfg.md_min_level,
            skip_heading_patterns=user_cfg.md_skip_heading_patterns or None,
            strip_heading_pattern=user_cfg.md_strip_heading_pattern or None,
        )
    return parse_pdf_to_tree(
        source_path,
        course=user_cfg.course_name,
        material=user_cfg.material_name,
        text_start_page=user_cfg.text_start_page,
        text_end_page=user_cfg.text_end_page,
        toc_re_expression=user_cfg.toc_re_expression,
        toc_max_level=user_cfg.toc_max_level,
    )


def run_phase1(
    source_path: str,
    user_cfg: UserConfig,
    api_cfg: ApiConfig,
    *,
    checkpoint_path: Optional[str] = None,
) -> KnowledgeGraph:
    logger.section(
        f"Phase 1 · 初始构建 (course={user_cfg.course_name}, source_type={user_cfg.source_type})"
    )

    logger.step(f"1) 解析 {user_cfg.source_type.upper()} + 章节切分")
    tree = parse_source_to_tree(source_path, user_cfg)
    diag = tree_summary(tree)
    logger.info(f"章节层级统计: {diag}")
    logger.info("大纲预览（前若干行）:\n" + render_outline(tree, max_lines=40))
    if diag.get("leaves", 0) == 0:
        logger.error(
            "切分后得到 0 个叶子节点。"
            + ("请检查 toc_re_expression 与 page 范围。" if user_cfg.source_type == "pdf"
               else "请检查 md_min_level/md_max_level 与文档标题。")
        )
        logger.pause("调整配置后回车继续，或 Ctrl+C 退出。")

    logger.step("2) 构造 KG 骨架")
    kg = build_kg_skeleton(tree)
    logger.info(f"KG 骨架: {kg.stats()}")

    llm = LLMClient(api_cfg)

    logger.step("3) 自下而上摘要")
    bottom_up_summarize(kg, llm, user_cfg.course_name)

    if checkpoint_path:
        kg.save(os.path.join(checkpoint_path, "phase1_after_summary.json"))
        logger.ok(f"已保存 checkpoint: phase1_after_summary.json")

    logger.step("4) 实体/关系抽取（叶子）")
    extract_entities_and_relations(kg, llm, user_cfg.course_name)

    logger.step("5) 章节关系抽取（中间层）")
    extract_section_relations(kg, llm, user_cfg.course_name)

    if checkpoint_path:
        kg.save(os.path.join(checkpoint_path, "phase1_done.json"))
        logger.ok("已保存 checkpoint: phase1_done.json")

    logger.ok(f"Phase 1 完成: {kg.stats()}")
    return kg
