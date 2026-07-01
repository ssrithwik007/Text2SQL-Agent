from app.core.engine import engine
from fastapi import APIRouter, HTTPException
from app.schemas.models import QueryRequest, QueryResponse, EmbedRequest, EmbedResponse
from app.services.retriever import execute_retrieval

router = APIRouter()

@router.post("/retrieve", response_model=QueryResponse)
async def retrieve_context(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    tables, columns, joins = execute_retrieval(
        query=request.query, 
        top_t=request.top_tables, 
        top_c=request.top_columns
    )
    
    if not request.get_sample_values:
        for col in columns:
            delattr(col, "sample_values")
    
    return QueryResponse(
        query=request.query,
        selected_tables=tables,
        selected_columns=columns,
        joins = joins
    )

@router.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    query_vector_np = engine.bi_model.encode(request.text, convert_to_numpy=True)

    return EmbedResponse(
        embeddings = query_vector_np.tolist()
    )