"""KG 数据结构。

参考论文 §3.1 的定义：
- 节点分层 V = V1 ∪ ... ∪ Vk
- 垂直边 E1：跨相邻层 (has_subsection / has_entity / has_subordinate)
- 水平边 E2：同层 (section_related / entity_related，附带具体 LLM 预测的子类型)
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


LAYER_BOOK = "book"
LAYER_CHAPTER_OR_SECTION = "section"  # TOC 中所有非叶子层级统一标 section（按 depth 区分）
LAYER_SUBSECTION_LEAF = "subsection"   # 叶子结点 (拥有正文)
LAYER_ENTITY_CORE = "core_entity"
LAYER_ENTITY_NONCORE = "non_core_entity"


VERT_HAS_SUBSECTION = "has_subsection"
VERT_HAS_ENTITY = "has_entity"
VERT_HAS_SUBORDINATE = "has_subordinate"
HORIZ_SECTION_RELATED = "section_related"
HORIZ_ENTITY_RELATED = "entity_related"


@dataclass
class Node:
    id: str
    name: str
    layer: str
    depth: int = 0  # 0=book, 1=chapter, ...
    description: str = ""
    raw_text: str = ""           # 仅 leaf TOC 节点存正文
    aliases: List[str] = field(default_factory=list)
    role: Optional[str] = None    # core / non-core (仅实体节点)
    embedding: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    kind: str             # vertical / horizontal
    category: str         # has_subsection / section_related / ...
    relation: str = ""   # LLM 预测的具体语义（horizontal）
    description: str = ""
    strength: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """简单内存图。"""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        # adjacency caches
        self._children: Dict[str, List[str]] = {}
        self._parent: Dict[str, str] = {}
        self._neighbors: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------- factory
    @staticmethod
    def new_id(prefix: str = "n") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}"

    # ----------------------------------------------------- mutation helpers
    def add_node(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise KeyError(f"重复节点 id: {node.id}")
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: Edge) -> Edge:
        if edge.src not in self.nodes or edge.dst not in self.nodes:
            raise KeyError(f"边引用了不存在的节点: {edge.src} -> {edge.dst}")
        self.edges.append(edge)
        if edge.kind == "vertical":
            self._children.setdefault(edge.src, []).append(edge.dst)
            self._parent[edge.dst] = edge.src
        self._neighbors.setdefault(edge.src, set()).add(edge.dst)
        self._neighbors.setdefault(edge.dst, set()).add(edge.src)
        return edge

    def remove_edge(self, edge: Edge) -> None:
        try:
            self.edges.remove(edge)
        except ValueError:
            return
        if edge.kind == "vertical":
            ch = self._children.get(edge.src)
            if ch and edge.dst in ch:
                ch.remove(edge.dst)
            if self._parent.get(edge.dst) == edge.src:
                self._parent.pop(edge.dst, None)
        # neighbors 重建仅当真的没有任何残留时才删
        if edge.dst in self._neighbors.get(edge.src, set()):
            still = any(
                (e.src == edge.src and e.dst == edge.dst) or (e.src == edge.dst and e.dst == edge.src)
                for e in self.edges
            )
            if not still:
                self._neighbors[edge.src].discard(edge.dst)
                self._neighbors[edge.dst].discard(edge.src)

    # ------------------------------------------------------------ queries
    def children(self, node_id: str) -> List[str]:
        return list(self._children.get(node_id, ()))

    def parent(self, node_id: str) -> Optional[str]:
        return self._parent.get(node_id)

    def neighbors(self, node_id: str) -> Set[str]:
        return set(self._neighbors.get(node_id, set()))

    def degree(self, node_id: str) -> int:
        return len(self._neighbors.get(node_id, ()))

    def nodes_at_layer(self, layer: str) -> List[Node]:
        return [n for n in self.nodes.values() if n.layer == layer]

    def leaf_toc_nodes(self) -> List[Node]:
        """没有 has_subsection 子节点的 TOC 节点（含 book 自身只在没有章节时才算）。"""
        toc_layers = {LAYER_BOOK, LAYER_CHAPTER_OR_SECTION, LAYER_SUBSECTION_LEAF}
        out: List[Node] = []
        for n in self.nodes.values():
            if n.layer not in toc_layers:
                continue
            kids = [c for c in self.children(n.id) if self.nodes[c].layer in toc_layers]
            if not kids:
                out.append(n)
        return out

    def lineage(self, node_id: str) -> List[str]:
        """从该节点回溯到根的 id 列表（包含自身），仅沿 vertical 边。"""
        path: List[str] = [node_id]
        cur = node_id
        while True:
            p = self._parent.get(cur)
            if p is None:
                break
            path.append(p)
            cur = p
        return path

    def common_ancestors(self, u: str, v: str) -> int:
        """论文 §3.1.4：两节点谱系路径交集大小（这里树结构下退化为 LCA 之上的祖先数）。"""
        pu = set(self.lineage(u))
        pv = set(self.lineage(v))
        return len(pu & pv)

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        kg = cls()
        for nd in data.get("nodes", []):
            kg.add_node(Node(**nd))
        for ed in data.get("edges", []):
            kg.add_edge(Edge(**ed))
        return kg

    # ------------------------------------------------------------- summary
    def stats(self) -> Dict[str, int]:
        out: Dict[str, int] = {"nodes": len(self.nodes), "edges": len(self.edges)}
        for n in self.nodes.values():
            out[f"layer:{n.layer}"] = out.get(f"layer:{n.layer}", 0) + 1
        for e in self.edges:
            out[f"edge:{e.category}"] = out.get(f"edge:{e.category}", 0) + 1
        return out
