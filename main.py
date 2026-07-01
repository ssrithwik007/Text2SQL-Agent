from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.engine import engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initializes the AI models before accepting web traffic
    engine.load(schema_path = "data/schema.json", 
                bi_path = "data/bi-encoder-metadata.json",
                ce_path = "data/cross-encoder-metadata.json",
                graph_path="data/graph.json")
    yield

app = FastAPI(title="Text-to-SQL API", lifespan=lifespan)
app.include_router(router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)