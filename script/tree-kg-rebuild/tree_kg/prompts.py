"""所有阶段使用的 Prompt 模板（中英混合）。

参考论文附录 B 的设计原则：Role / Task / Constraints / Output Template / Example。
为了简化，这里统一使用 system+user 两段，user 中嵌入数据。
"""

from __future__ import annotations

from typing import Any, Dict, List


def _system(role: str, task: str, constraints: List[str], output_template: str) -> str:
    parts = [
        f"# 角色\n{role}",
        f"# 任务\n{task}",
        "# 约束\n" + "\n".join(f"- {c}" for c in constraints),
        f"# 输出格式\n```json\n{output_template}\n```\n严格输出合法 JSON，不要包含解释性文字。",
    ]
    return "\n\n".join(parts)


# ---------------------------------------------------------------- summary

def summary_messages(course: str, parent_summary: str, leaf_text: str) -> List[Dict[str, str]]:
    system = _system(
        role=f"你是 {course} 领域的资深教学专家，擅长抽取教材小节要点。",
        task="基于给定的最小章节正文，输出一段紧凑、术语丰富、覆盖核心知识点的小节摘要。",
        constraints=[
            "保留学科术语和定义性陈述",
            "不要复述提示词、上下文或元信息",
            "200~400 字，以中文输出",
            "尽量列出关键名词、量、规则、定理、公式、典型现象等",
        ],
        output_template='{"summary": "..."}',
    )
    user = (
        f"## 上级章节摘要（仅做语境，可为空）\n{parent_summary or '(无)'}\n\n"
        f"## 当前章节正文\n{leaf_text.strip()[:6000]}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def aggregate_summary_messages(course: str, child_summaries: List[str]) -> List[Dict[str, str]]:
    system = _system(
        role=f"你是 {course} 领域的资深教学专家。",
        task="将给定的若干子章节摘要聚合，输出当前父章节的整体摘要。",
        constraints=[
            "保留所有重要术语和子主题",
            "200~500 字，中文输出",
            "结构清晰、突出主题脉络",
        ],
        output_template='{"summary": "..."}',
    )
    blob = "\n\n".join(f"### 子章节 {i + 1}\n{s}" for i, s in enumerate(child_summaries))
    return [{"role": "system", "content": system}, {"role": "user", "content": blob}]


# ----------------------------------------------------------- entity extract

def entity_extract_messages(course: str, section_path: str, summary: str) -> List[Dict[str, str]]:
    system = _system(
        role=f"你是 {course} 领域的术语抽取专家。",
        task="从小节摘要中抽取与学科强相关的实体（名词或名词短语）。",
        constraints=[
            "实体必须简洁、具体、强领域相关；忽略示例性、生活化的词",
            "同一实体的不同写法合并入 alias",
            "type 取自：concept / law / formula / phenomenon / quantity / device / method / other",
            "raw_content 直接引自摘要，便于追溯",
            "输出合法 JSON 对象",
        ],
        output_template='{"entities": [{"name": "...", "alias": ["..."], "type": "...", "raw_content": "..."}]}',
    )
    user = f"## 章节路径\n{section_path}\n\n## 摘要\n{summary}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def relation_extract_messages(
    course: str,
    section_path: str,
    summary: str,
    entities: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    system = _system(
        role=f"你是 {course} 领域的关系抽取专家。",
        task="基于小节摘要和给定实体列表，抽取实体之间真实存在的语义关系。",
        constraints=[
            "只抽取摘要中能找到证据的关系，不要凭空想象",
            "type 简短具名（动词/动词短语，例如 obey / produces / depends_on / has）",
            "strength 0-10，越能从文本支持越高",
            "subject 与 object 必须取自给出的实体名 (优先使用 name)",
            "输出合法 JSON",
        ],
        output_template='{"relations": [{"subject": "...", "object": "...", "type": "...", "description": "...", "strength": 0}]}',
    )
    ent_blob = "\n".join(
        f"- {e.get('name')}（{e.get('type', '')}）" for e in entities
    )
    user = f"## 章节路径\n{section_path}\n\n## 摘要\n{summary}\n\n## 候选实体\n{ent_blob}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def section_relation_messages(
    course: str,
    parent_path: str,
    parent_summary: str,
    children: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    system = _system(
        role=f"你是 {course} 领域的教材结构分析专家。",
        task="给定父章节摘要与其多个子章节摘要，找出子章节之间存在的语义/学理关系。",
        constraints=[
            "subject/object 必须使用给定的子章节名",
            "type 用动词或动词短语表达",
            "strength 0-10",
            "如果不存在关系可输出空数组",
            "输出合法 JSON",
        ],
        output_template='{"relations": [{"subject": "...", "object": "...", "type": "...", "description": "...", "strength": 0}]}',
    )
    blob = "\n".join(f"### {c['name']}\n{c['summary']}" for c in children)
    user = f"## 父章节路径\n{parent_path}\n\n## 父章节摘要\n{parent_summary}\n\n## 子章节摘要\n{blob}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ----------------------------------------------------------- conv (描述增强)

def conv_messages(course: str, entity: Dict[str, Any], neighbors: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = _system(
        role=f"你是 {course} 领域专家，擅长基于上下文增强实体描述。",
        task="基于实体已有信息和邻居信息，生成一份结构化的增强描述报告。",
        constraints=[
            "definition 一句话，给出标准定义",
            "description 详细解释，含背景与重要属性",
            "role 描述该实体在邻居子图中的角色",
            "中文输出，避免冗余",
            "输出合法 JSON",
        ],
        output_template='{"definition": "...", "description": "...", "role": "..."}',
    )
    nb = "\n".join(
        f"- {n['name']}: {n.get('description', '')[:200]}" for n in neighbors[:30]
    )
    user = (
        f"## 实体\n名称: {entity['name']}\n类型: {entity.get('type', '')}\n原描述: {entity.get('description', '')}\n\n"
        f"## 邻居（最多 30 个）\n{nb or '(无)'}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ----------------------------------------------------------- aggr (核心判定)

def aggr_role_messages(course: str, entity: Dict[str, Any], neighbors: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = _system(
        role=f"你是 {course} 领域专家，擅长判断实体在子图中的角色。",
        task="判断当前实体是 core（核心实体，可作为多个外延概念的中心）还是 non-core（外延、附属概念）。",
        constraints=[
            "core: 教材中作为重要主题或被反复指代",
            "non-core: 仅作为某 core 的子例/现象/特例",
            "输出合法 JSON",
        ],
        output_template='{"role": "core|non-core", "reason": "..."}',
    )
    nb = "\n".join(
        f"- {n['name']}: {n.get('description', '')[:120]}" for n in neighbors[:20]
    )
    user = (
        f"## 实体\n名称: {entity['name']}\n描述: {entity.get('description', '')}\n\n"
        f"## 邻居\n{nb or '(无)'}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ----------------------------------------------------------- dedup

def dedup_messages(course: str, e1: Dict[str, Any], e2: Dict[str, Any]) -> List[Dict[str, str]]:
    system = _system(
        role=f"你是 {course} 领域专家。",
        task="判断给定的两个候选实体是否表示同一概念。",
        constraints=[
            "考虑专有写法、别名、近义术语",
            "若仅是相关而非同一，输出 false",
            "输出合法 JSON",
        ],
        output_template='{"same": true, "reason": "..."}',
    )
    user = (
        f"## 实体 A\n名称: {e1['name']}\n别名: {e1.get('aliases', [])}\n描述: {e1.get('description', '')}\n\n"
        f"## 实体 B\n名称: {e2['name']}\n别名: {e2.get('aliases', [])}\n描述: {e2.get('description', '')}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ----------------------------------------------------------- pred

def edge_pred_messages(course: str, e1: Dict[str, Any], e2: Dict[str, Any]) -> List[Dict[str, str]]:
    system = _system(
        role=f"你是 {course} 领域专家。",
        task="判断两个实体之间是否存在有意义的关系，并给出强度评分。",
        constraints=[
            "is_relevant: 是否有显著关系",
            "type: 关系名（动词或动词短语，简短）",
            "strength: 0-10，越紧密越高",
            "description: 简明解释，必须基于学科内在逻辑",
            "输出合法 JSON",
        ],
        output_template='{"is_relevant": true, "type": "...", "strength": 0, "description": "..."}',
    )
    user = (
        f"## 实体 A\n名称: {e1['name']}\n描述: {e1.get('description', '')}\n\n"
        f"## 实体 B\n名称: {e2['name']}\n描述: {e2.get('description', '')}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
