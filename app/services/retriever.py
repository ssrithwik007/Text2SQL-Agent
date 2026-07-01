import torch
import faiss
import networkx as nx
from itertools import combinations
from app.core.engine import engine
from sentence_transformers import util
from app.schemas.models import ColumnMatch

def execute_retrieval(query: str, top_t: int = 5, top_c: int = 12):
    instruction = "Represent this sentence for searching relevant passages: "
    formatted_query = instruction + query
    
    # 1. FAISS Table Search
    query_vector_np = engine.bi_model.encode([formatted_query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector_np)
    
    distances, indices = engine.faiss_index.search(query_vector_np, k=top_t)
    top_tables = [engine.table_name_map[idx] for idx in indices[0] if idx != -1]

    # 2. Relationship retrieval
    retrieved_tables = set()
    edges = set()
    joins = []

    for src, dst in combinations(top_tables, 2):
        try:
            path = nx.shortest_path(engine.graph, src, dst)

            retrieved_tables.update(path)

            for u, v in zip(path, path[1:]):
                edges.add(tuple(sorted((u, v))))

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            retrieved_tables.add(src)
            retrieved_tables.add(dst)

    for u, v in edges:
        edge_data = engine.graph.get_edge_data(u, v)

        for join in edge_data["join"]:
            joins.append(
                f"{join['left_column']} joins {join['right_column']}"
            )

    top_tables = list(retrieved_tables)
    
    # 3. Fast Column Search
    valid_tensors = []
    valid_metadata = []
    
    for table in top_tables:
        if table in engine.precomputed_col_vectors:
            valid_tensors.append(engine.precomputed_col_vectors[table])
            valid_metadata.extend(engine.precomputed_col_metadata[table])
            
    if not valid_tensors:
        return top_tables, []
        
    search_space = torch.cat(valid_tensors, dim=0)
    query_vector_tensor = engine.bi_model.encode(formatted_query, convert_to_tensor=True)
    
    hits = util.semantic_search(query_vector_tensor, search_space, top_k=top_c)[0]
    
    selected_columns = []
    for hit in hits:
        matched_col = valid_metadata[hit['corpus_id']]
        selected_columns.append(ColumnMatch(
            table=matched_col["table"],
            column=matched_col["column"],
            description=matched_col["description"],
            sample_values=matched_col["sample_values"],
            score=round(hit['score'], 4)
        ))
        
    return top_tables, selected_columns, joins