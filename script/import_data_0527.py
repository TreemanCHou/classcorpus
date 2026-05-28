# Import data to neo4j. 
# version 0.0

import os
import json
from collections import defaultdict
from neo4j import GraphDatabase

# ==========================
# 配置你的 Neo4j 数据库连接
# ==========================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "B1B2B3B4"  # 请修改为你的实际密码

def import_to_neo4j():
    # 确保使用绝对路径读取 json 文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "tree-kg-rebuild", "output", "chemi_0526_v1.json")
    
    print(f"正在读取数据文件: {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    print(f"共发现 {len(nodes)} 个节点，{len(edges)} 条边。")

    # 建立 Neo4j 连接
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # 1. 创建唯一性约束，这能极大加快边的连接速度并防止重复
        print("正在创建节点约束...")
        try:
            # 兼容 Neo4j 5.x 语法
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:KnowledgeNode) REQUIRE n.id IS UNIQUE")
        except Exception as e:
            pass
        
        # 2. 分批导入节点 (批量提速)
        print("开始导入节点...")
        node_query = """
        UNWIND $batch AS node
        MERGE (n:KnowledgeNode {id: node.id})
        SET n.name = node.name,
            n.layer = node.layer,
            n.depth = node.depth,
            n.description = node.description,
            n.embedding = node.embedding
        """
        batch_size = 5000
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i+batch_size]
            session.run(node_query, batch=batch)
            print(f"  已导入 {min(i+batch_size, len(nodes))}/{len(nodes)} 个节点")

        # 3. 按关系类型对边进行分组
        # Cypher 不支持直接把 relationship_type 当参数传，需要单独切分
        print("正在按关系类型处理边...")
        edges_by_category = defaultdict(list)
        for edge in edges:
            cat = edge.get("category")
            if not cat:
                cat = "RELATED_TO"
            # Neo4j 关系类型建议大写且不包含短横线
            cat = str(cat).upper().replace(" ", "_").replace("-", "_")
            edges_by_category[cat].append(edge)
            
        # 4. 分批导入边
        print("开始导入边 (关系)...")
        for rel_type, rel_edges in edges_by_category.items():
            print(f"  导入关系 [{rel_type}]，共 {len(rel_edges)} 条...")
            
            edge_query = f"""
            UNWIND $batch AS edge
            MATCH (src:KnowledgeNode {{id: edge.src}})
            MATCH (dst:KnowledgeNode {{id: edge.dst}})
            MERGE (src)-[r:{rel_type}]->(dst)
            SET r.kind = edge.kind,
                r.strength = edge.strength
            """
            for i in range(0, len(rel_edges), batch_size):
                batch = rel_edges[i:i+batch_size]
                session.run(edge_query, batch=batch)

    driver.close()
    print("Neo4j 数据导入成功完成！")

if __name__ == "__main__":
    import_to_neo4j()