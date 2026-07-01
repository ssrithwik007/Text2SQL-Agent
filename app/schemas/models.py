from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    top_tables: int = 5
    top_columns: int = 12
    get_sample_values: bool = False

class ColumnMatch(BaseModel):
    table: str
    column: str
    description: str
    sample_values: list | None = None
    score: float

class QueryResponse(BaseModel):
    query: str
    selected_tables: list[str]
    selected_columns: list[ColumnMatch]
    joins: list[str]

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embeddings: list