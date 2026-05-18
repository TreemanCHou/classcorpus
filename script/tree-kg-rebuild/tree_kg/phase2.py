"""Phase 2: 迭代扩展。

实现操作子：
- conv  : 上下文卷积，增强实体描述（论文 §3.3.1）
- aggr  : core/non-core 角色判定 + horizontal -> has_subordinate 转换（§3.3.2）
- embed : 对节点描述做向量化（§3.3.3）
- dedup : FAISS/最近邻 + LLM 判同 + 并查集合并（§3.3.4）
- pred  : 两阶段边预测 score = α·cos + β·AA + γ·CA（§3.3.5）
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from . import logger, prompts
from .config import ApiConfig
from .graph import (
    Edge,
    HORIZ_ENTITY_RELATED,
    KnowledgeGraph,
    LAYER_ENTITY_CORE,
    LAYER_ENTITY_NONCORE,
    LAYER_SUBSECTION_LEAF,
    Node,
    VERT_HAS_SUBORDINATE,
)
from .llm import LLMClient


def _entity_layers() -> Set[str]:
    return {LAYER_ENTITY_CORE, LAYER_ENTITY_NONCORE}


# ----------------------------------------------------------------- 1) conv

def run_conv(kg: KnowledgeGraph, llm: LLMClient, course: str) -> None:
    """对每个实体节点：基于其邻居增强描述。

    一步即可，论文表5 显示 conv 1 步收敛。
    """
    targets = [n for n in kg.nodes.values() if n.layer in _entity_layers()]
    logger.subsection(f"Phase2.A · 上下文卷积 conv（{len(targets)} 个实体）")
    for ent in tqdm(targets, desc="conv", unit="ent"):
        nb_ids = list(kg.neighbors(ent.id))
        nbs = [
            {"name": kg.nodes[i].name, "description": kg.nodes[i].description, "type": kg.nodes[i].meta.get("type", "")}
            for i in nb_ids if kg.nodes[i].layer in _entity_layers() or kg.nodes[i].layer.startswith("subsection")
        ]
        try:
            res = llm.chat_json(
                "conv",
                prompts.conv_messages(course, {
                    "name": ent.name,
                    "type": ent.meta.get("type", ""),
                    "description": ent.description,
                }, nbs),
            )
            if isinstance(res, dict):
                desc = res.get("description") or ""
                definition = res.get("definition") or ""
                role = res.get("role") or ""
                merged = "\n".join(filter(None, [
                    f"【定义】{definition}" if definition else "",
                    f"【说明】{desc}" if desc else "",
                    f"【角色】{role}" if role else "",
                ]))
                if merged:
                    ent.description = merged
        except Exception as e:
            logger.warn(f"conv 失败 {ent.name[:30]}: {e}")
            logger.pause("回车继续。")


# -------------------------------------------------------- 2) entity aggr

def run_aggr(kg: KnowledgeGraph, llm: LLMClient, course: str) -> None:
    """判定核心/非核心，把非核心实体的水平边转为 has_subordinate 垂直边。"""
    targets = [n for n in kg.nodes.values() if n.layer in _entity_layers()]
    logger.subsection(f"Phase2.B · 实体聚合 aggr（{len(targets)} 个实体）")

    # role classification
    for ent in tqdm(targets, desc="role", unit="ent"):
        nbs = []
        for nb_id in kg.neighbors(ent.id):
            nb = kg.nodes[nb_id]
            if nb.layer in _entity_layers():
                nbs.append({"name": nb.name, "description": nb.description})
        try:
            res = llm.chat_json("aggr", prompts.aggr_role_messages(course, {
                "name": ent.name, "description": ent.description,
            }, nbs))
            role = (res.get("role") if isinstance(res, dict) else "").strip().lower()
            if role in ("core", "non-core"):
                ent.role = role
                ent.layer = LAYER_ENTITY_CORE if role == "core" else LAYER_ENTITY_NONCORE
        except Exception as e:
            logger.warn(f"aggr 角色判定失败 {ent.name[:30]}: {e}")
            logger.pause("回车继续。")

    # transform horizontal -> vertical for non-core peripherals
    logger.info("将非核心实体的水平连接重构为 has_subordinate 垂直边……")
    converted = 0
    to_remove: List[Edge] = []
    for e in list(kg.edges):
        if e.kind != "horizontal" or e.category != HORIZ_ENTITY_RELATED:
            continue
        a = kg.nodes.get(e.src)
        b = kg.nodes.get(e.dst)
        if not a or not b:
            continue
        if a.layer == LAYER_ENTITY_CORE and b.layer == LAYER_ENTITY_NONCORE:
            core, peri = a, b
        elif b.layer == LAYER_ENTITY_CORE and a.layer == LAYER_ENTITY_NONCORE:
            core, peri = b, a
        else:
            continue
        # 若 peri 已有其他垂直父，则跳过（避免树形破坏）
        if kg.parent(peri.id) and kg.nodes[kg.parent(peri.id)].layer != LAYER_SUBSECTION_LEAF:
            continue
        kg.add_edge(Edge(
            src=core.id, dst=peri.id, kind="vertical",
            category=VERT_HAS_SUBORDINATE,
            relation=e.relation, description=e.description, strength=e.strength,
        ))
        to_remove.append(e)
        converted += 1
    for e in to_remove:
        kg.remove_edge(e)
    logger.info(f"已转换 {converted} 条 entity_related -> has_subordinate")


# -------------------------------------------------------------- 3) embed

def run_embed(kg: KnowledgeGraph, llm: LLMClient) -> Dict[str, np.ndarray]:
    """对所有节点描述做嵌入，归一化后写入 node.embedding。"""
    nodes = list(kg.nodes.values())
    logger.subsection(f"Phase2.C · 节点嵌入 embed（{len(nodes)} 个节点）")
    inputs: List[str] = []
    indices: List[str] = []
    for n in nodes:
        text = (n.name or "") + "\n" + (n.description or "")
        if not text.strip():
            n.embedding = None
            continue
        inputs.append(text[:6000])
        indices.append(n.id)

    if not inputs:
        return {}

    vecs = llm.embed("embed", inputs)
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms

    out: Dict[str, np.ndarray] = {}
    for i, nid in enumerate(indices):
        kg.nodes[nid].embedding = arr[i].tolist()
        out[nid] = arr[i]
    logger.ok(f"嵌入完成，维度 = {arr.shape[1]}")
    return out


def _gather_entity_embeddings(kg: KnowledgeGraph) -> Tuple[List[str], np.ndarray]:
    ids: List[str] = []
    vecs: List[List[float]] = []
    for n in kg.nodes.values():
        if n.layer in _entity_layers() and n.embedding is not None:
            ids.append(n.id)
            vecs.append(n.embedding)
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    return ids, np.asarray(vecs, dtype=np.float32)


# -------------------------------------------------------------- 4) dedup

def run_dedup(
    kg: KnowledgeGraph,
    llm: LLMClient,
    course: str,
    threshold: float = 0.55,
    top_k: int = 20,
) -> int:
    """实体去重：FAISS/最近邻 + LLM 判同 + 并查集合并。

    - 我们使用归一化向量上的 L2 距离 + sklearn NearestNeighbors 替代 FAISS（更轻量、跨平台）。
    - 仅在同 role 下合并。
    """
    ids, vecs = _gather_entity_embeddings(kg)
    logger.subsection(f"Phase2.D · 实体去重 dedup（{len(ids)} 个实体）")
    if len(ids) < 2:
        return 0

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        logger.error(f"缺少 sklearn: {e}")
        return 0

    k = min(top_k + 1, len(ids))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(vecs)
    dist, idx = nn.kneighbors(vecs)

    # 候选对（去重 + 过滤 role + 距离阈值）
    parent: Dict[str, str] = {nid: nid for nid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    candidates: List[Tuple[float, str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for i, src_id in enumerate(ids):
        src_role = kg.nodes[src_id].role
        for j in range(1, k):  # skip self
            d = float(dist[i, j])
            if d >= threshold:
                continue
            tgt_id = ids[idx[i, j]]
            if tgt_id == src_id:
                continue
            if kg.nodes[tgt_id].role != src_role:
                continue
            key = tuple(sorted((src_id, tgt_id)))
            if key in seen:
                continue
            seen.add(key)
            candidates.append((d, key[0], key[1]))
    candidates.sort(key=lambda x: x[0])
    logger.info(f"候选对数 = {len(candidates)}")

    merged_count = 0
    for d, a, b in tqdm(candidates, desc="dedup", unit="pair"):
        if find(a) == find(b):
            continue
        try:
            res = llm.chat_json("dedup", prompts.dedup_messages(course, {
                "name": kg.nodes[a].name,
                "aliases": kg.nodes[a].aliases,
                "description": kg.nodes[a].description,
            }, {
                "name": kg.nodes[b].name,
                "aliases": kg.nodes[b].aliases,
                "description": kg.nodes[b].description,
            }))
            same = bool(res.get("same")) if isinstance(res, dict) else False
        except Exception as e:
            logger.warn(f"dedup 判同失败 {kg.nodes[a].name} ~ {kg.nodes[b].name}: {e}")
            logger.pause("回车继续，将跳过本对。")
            continue
        if same and union(a, b):
            merged_count += 1

    if not merged_count:
        return 0

    # 物理合并：每个集合保留 representative，把其它节点的所有边迁移过去后删除
    groups: Dict[str, List[str]] = {}
    for nid in ids:
        groups.setdefault(find(nid), []).append(nid)

    for rep, members in groups.items():
        if len(members) <= 1:
            continue
        keep = rep
        for nid in members:
            if nid == keep:
                continue
            _merge_entity_into(kg, src_id=nid, keep_id=keep)

    logger.ok(f"dedup 合并 {merged_count} 对实体")
    return merged_count


def _merge_entity_into(kg: KnowledgeGraph, *, src_id: str, keep_id: str) -> None:
    """把 src_id 的所有边迁移到 keep_id 上，并删除 src 节点。"""
    if src_id not in kg.nodes or keep_id not in kg.nodes:
        return
    src = kg.nodes[src_id]
    keep = kg.nodes[keep_id]
    keep.aliases = list(dict.fromkeys((keep.aliases or []) + [src.name] + (src.aliases or [])))
    if src.description and len(src.description) > len(keep.description):
        keep.description = src.description

    new_edges: List[Edge] = []
    for e in kg.edges:
        if e.src == src_id and e.dst == keep_id:
            continue  # 自环，丢弃
        if e.dst == src_id and e.src == keep_id:
            continue
        if e.src == src_id:
            new_edges.append(Edge(
                src=keep_id, dst=e.dst, kind=e.kind, category=e.category,
                relation=e.relation, description=e.description, strength=e.strength,
                meta=dict(e.meta),
            ))
        elif e.dst == src_id:
            new_edges.append(Edge(
                src=e.src, dst=keep_id, kind=e.kind, category=e.category,
                relation=e.relation, description=e.description, strength=e.strength,
                meta=dict(e.meta),
            ))
        else:
            new_edges.append(e)

    # rebuild graph caches
    kg.edges = []
    kg._children = {}
    kg._parent = {}
    kg._neighbors = {}
    for e in new_edges:
        try:
            kg.add_edge(e)
        except KeyError:
            continue

    kg.nodes.pop(src_id, None)


# -------------------------------------------------------------- 5) pred

def _adamic_adar(kg: KnowledgeGraph, u: str, v: str) -> float:
    nu = kg.neighbors(u)
    nv = kg.neighbors(v)
    common = nu & nv
    s = 0.0
    for w in common:
        d = kg.degree(w)
        if d > 1:
            s += 1.0 / math.log(d)
    return s


def run_pred(
    kg: KnowledgeGraph,
    llm: LLMClient,
    course: str,
    *,
    alpha: float = 0.6,
    beta_stage1: float = 0.0,
    gamma_stage1: float = 0.4,
    beta_stage2: float = 0.3,
    gamma_stage2: float = 0.1,
    strength_threshold: int = 6,
    delta_e_each_stage: int = 500,
    max_candidates: int = 5000,
    top_k_neighbors: int = 30,
) -> int:
    """两阶段边预测。

    - Stage 1: β=0, γ=0.4 仅用语义+共同祖先增强连通性
    - Stage 2: β=0.3, γ=0.1 完整公式
    """
    ids, vecs = _gather_entity_embeddings(kg)
    if len(ids) < 2:
        return 0
    logger.subsection(f"Phase2.E · 边预测 pred（{len(ids)} 个实体）")

    # 候选对：每个实体取语义最相近 top-K 内为候选；过滤已存在边和距离过大
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        logger.error(f"缺少 sklearn: {e}")
        return 0
    k = min(top_k_neighbors + 1, len(ids))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(vecs)
    cos_dist, idx = nn.kneighbors(vecs)
    cos_sim = 1.0 - cos_dist

    pos_of = {nid: i for i, nid in enumerate(ids)}

    def existing_edge(u: str, v: str) -> bool:
        ns = kg.neighbors(u)
        return v in ns

    def collect_candidates(beta: float, gamma: float, exclude_same_subsection: bool) -> List[Tuple[float, str, str]]:
        out: List[Tuple[float, str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        for i, src_id in enumerate(ids):
            src_subs = _subsection_of(kg, src_id)
            for j in range(1, k):
                tgt_id = ids[idx[i, j]]
                if tgt_id == src_id:
                    continue
                if existing_edge(src_id, tgt_id):
                    continue
                if exclude_same_subsection and src_subs and src_subs == _subsection_of(kg, tgt_id):
                    continue
                key = tuple(sorted((src_id, tgt_id)))
                if key in seen:
                    continue
                seen.add(key)
                cs = float(cos_sim[i, j])
                aa = _adamic_adar(kg, src_id, tgt_id) if beta > 0 else 0.0
                ca = float(kg.common_ancestors(src_id, tgt_id))
                score = alpha * cs + beta * aa + gamma * ca
                out.append((score, key[0], key[1]))
        out.sort(key=lambda x: -x[0])
        return out[:max_candidates]

    added_total = 0

    def evaluate_and_add(cands: List[Tuple[float, str, str]], budget: int, label: str) -> int:
        added = 0
        bar = tqdm(cands, desc=label, unit="pair")
        for score, a, b in bar:
            if added >= budget:
                break
            if existing_edge(a, b):
                continue
            try:
                res = llm.chat_json("pred", prompts.edge_pred_messages(course, {
                    "name": kg.nodes[a].name, "description": kg.nodes[a].description,
                }, {
                    "name": kg.nodes[b].name, "description": kg.nodes[b].description,
                }))
                if not isinstance(res, dict):
                    continue
                if not res.get("is_relevant"):
                    continue
                strength = float(res.get("strength", 0))
                if strength < strength_threshold:
                    continue
                kg.add_edge(Edge(
                    src=a, dst=b, kind="horizontal", category=HORIZ_ENTITY_RELATED,
                    relation=str(res.get("type", "")),
                    description=str(res.get("description", "")),
                    strength=strength,
                    meta={"score": score, "stage": label},
                ))
                added += 1
                bar.set_postfix(added=added)
            except Exception as e:
                logger.warn(f"pred 调用失败: {e}")
                logger.pause("回车继续。")
        return added

    logger.info(f"Stage 1: alpha={alpha}, beta={beta_stage1}, gamma={gamma_stage1}")
    cands1 = collect_candidates(beta_stage1, gamma_stage1, exclude_same_subsection=False)
    added_total += evaluate_and_add(cands1, delta_e_each_stage, "pred-s1")

    logger.info(f"Stage 2: alpha={alpha}, beta={beta_stage2}, gamma={gamma_stage2}")
    cands2 = collect_candidates(beta_stage2, gamma_stage2, exclude_same_subsection=True)
    added_total += evaluate_and_add(cands2, delta_e_each_stage, "pred-s2")

    logger.ok(f"pred 共新增 {added_total} 条边")
    return added_total


def _subsection_of(kg: KnowledgeGraph, entity_id: str) -> Optional[str]:
    """向上回溯找到承载实体的 subsection 节点 id（若有）。"""
    cur = entity_id
    while cur:
        n = kg.nodes.get(cur)
        if n is None:
            return None
        if n.layer == LAYER_SUBSECTION_LEAF:
            return cur
        p = kg.parent(cur)
        if p is None or p == cur:
            return None
        cur = p
    return None


# ------------------------------------------------------------------ driver

def run_phase2(
    kg: KnowledgeGraph,
    api_cfg: ApiConfig,
    course: str,
    *,
    checkpoint_path: Optional[str] = None,
) -> KnowledgeGraph:
    logger.section(f"Phase 2 · 迭代扩展 (course={course})")
    llm = LLMClient(api_cfg)
    rt = api_cfg.runtime

    logger.step("1) 上下文卷积 conv")
    run_conv(kg, llm, course)
    if checkpoint_path:
        kg.save(os.path.join(checkpoint_path, "phase2_after_conv.json"))

    logger.step("2) 实体聚合 aggr")
    run_aggr(kg, llm, course)
    if checkpoint_path:
        kg.save(os.path.join(checkpoint_path, "phase2_after_aggr.json"))

    logger.step("3) 节点嵌入 embed")
    run_embed(kg, llm)
    if checkpoint_path:
        kg.save(os.path.join(checkpoint_path, "phase2_after_embed.json"))

    logger.step("4) 实体去重 dedup")
    run_dedup(kg, llm, course, threshold=rt.dedup_threshold)
    # 去重后需要重新嵌入，因为有节点被合并，可选；这里仅在尺寸有较大变化时再做
    if checkpoint_path:
        kg.save(os.path.join(checkpoint_path, "phase2_after_dedup.json"))

    logger.step("5) 边预测 pred")
    extra = max(rt.edge_pred_extra_edges, 1)
    run_pred(
        kg, llm, course,
        alpha=rt.edge_pred_alpha,
        beta_stage1=rt.edge_pred_beta_stage1,
        gamma_stage1=rt.edge_pred_gamma_stage1,
        beta_stage2=rt.edge_pred_beta_stage2,
        gamma_stage2=rt.edge_pred_gamma_stage2,
        strength_threshold=rt.edge_pred_strength_threshold,
        delta_e_each_stage=max(extra // 2, 1),
    )

    if checkpoint_path:
        kg.save(os.path.join(checkpoint_path, "phase2_done.json"))
        logger.ok("已保存 checkpoint: phase2_done.json")

    logger.ok(f"Phase 2 完成: {kg.stats()}")
    return kg
