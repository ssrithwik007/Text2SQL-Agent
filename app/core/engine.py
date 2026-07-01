import json
import faiss
from networkx.readwrite import json_graph
from sentence_transformers import SentenceTransformer, CrossEncoder

class Engine:
    def __init__(self):
        self.bi_model = None
        self.cross_encoder = None
        self.faiss_index = None
        self.graph = None

        self.schema = {}            # Full table/column details
        self.bi_metadata = {}       # {table: dense_text}
        self.ce_metadata = {}       # {table: rich_text_for_reranking}
        self.table_name_map = []    # Index to Table Name lookup
        self.graph_data = []

        self.precomputed_col_vectors = {}
        self.precomputed_col_metadata = {}

    def load(self, schema_path, bi_path, ce_path, graph_path):
        print("Loading models and metadata...")
        self.bi_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        
        # Load your three specific files
        with open(schema_path, "r") as f: self.schema = json.load(f)
        with open(bi_path, "r") as f: self.bi_metadata = json.load(f)
        with open(ce_path, "r") as f: self.ce_metadata = json.load(f)
        with open(graph_path, "r") as f: self.graph_data = json.load(f)
            
        # Build FAISS index from bi_metadata (Dense)
        self.table_name_map = list(self.bi_metadata.keys())
        table_texts = list(self.bi_metadata.values())
        
        print("Building FAISS Index...")
        embeddings = self.bi_model.encode(table_texts, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)

        print("Vectorizing columns for fast lookup...")
        for table_name, table_info in self.schema.items():
            columns = table_info.get("columns", {})
            if not columns: continue
            
            col_names = list(columns.keys())
            col_descriptions = [meta.get("description", "") for meta in columns.values()]
            col_samples = [meta.get("sample_values", "") for meta in columns.values()]
            
            search_texts = [
                f"Table: {table_name} | Column: {name} | Description: {desc}"
                for name, desc in zip(col_names, col_descriptions)
            ]
            
            # Store in RAM for instant retrieval
            self.precomputed_col_vectors[table_name] = self.bi_model.encode(search_texts, convert_to_tensor=True)
            self.precomputed_col_metadata[table_name] = [
                {"table": table_name, "column": name, "description": desc, "sample_values": samples} 
                for name, desc, samples in zip(col_names, col_descriptions, col_samples)
            ]

        print("Building graph for relationship retrieval...")
        self.graph = json_graph.node_link_graph(self.graph_data)

engine = Engine()