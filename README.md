# Text2SQL Agent

Text2SQL Agent is a natural-language-to-SQL system that helps translate a user's question into a database query using a retrieval-augmented pipeline. The project combines a FastAPI service, Hugging Face embedding and reranking models, and an n8n workflow that orchestrates query caching, context retrieval, SQL generation, validation, and execution.

## What It Does

The system is designed to accept a plain-English question such as "show me the top customers by revenue last month" and turn it into a valid PostgreSQL query. It does this by:

1. Encoding the user query with a Hugging Face bi-encoder.
2. Retrieving the most relevant tables, columns, and joins from precomputed schema metadata.
3. Passing that context into an n8n workflow that generates SQL with an LLM.
4. Validating and executing the SQL against PostgreSQL.
5. Returning the final response through the workflow's webhook response.

The FastAPI service also exposes a dedicated embedding endpoint so the n8n workflow can request query vectors for caching and similarity search.

## Architecture

- FastAPI service: serves retrieval and embedding endpoints.
- Sentence Transformers bi-encoder: creates dense embeddings for questions and schema metadata.
- Cross-encoder metadata: supports richer reranking context.
- FAISS: performs fast nearest-neighbor lookup over table embeddings.
- NetworkX graph: stores table relationships and join paths.
- n8n workflow: orchestrates cache lookup, schema retrieval, SQL generation, execution, and response formatting.
- PostgreSQL: stores cached query-to-SQL mappings and executes generated SQL.

## Repository Layout

```text
main.py                     # FastAPI application entrypoint
app/
	api/routes.py             # API routes for retrieval and embedding
	core/engine.py            # Model loading and retrieval engine setup
	schemas/models.py         # Pydantic request/response models
	services/retriever.py     # Table, column, and join retrieval logic
data/                       # Schema and metadata used by the retriever
n8n-workflow.json           # n8n workflow export
requirements.txt            # Python dependencies
```

## Prerequisites

- Python 3.10+ recommended
- PostgreSQL instance for the n8n workflow cache and query execution
- n8n instance for importing and running the workflow
- Internet access on first run so Hugging Face models can be downloaded

## Installation

Create and activate a virtual environment, then install the Python dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you are using PowerShell and the execution policy blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Running the API

Start the FastAPI app with Uvicorn:

```bash
uvicorn main:app --reload
```

On startup, the app loads:

- `data/schema.json`
- `data/bi-encoder-metadata.json`
- `data/cross-encoder-metadata.json`
- `data/graph.json`

The initial load will download and initialize:

- `BAAI/bge-base-en-v1.5`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`

## API Endpoints

### `POST /retrieve`

Retrieves the most relevant tables, columns, and join hints for a natural-language question.

Request body:

```json
{
	"query": "show total sales by month",
	"top_tables": 5,
	"top_columns": 12,
	"get_sample_values": false
}
```

Response shape:

```json
{
	"query": "show total sales by month",
	"selected_tables": ["orders", "customers"],
	"selected_columns": [
		{
			"table": "orders",
			"column": "order_date",
			"description": "Date when the order was created",
			"sample_values": ["2024-01-01"],
			"score": 0.9123
		}
	],
	"joins": ["orders.customer_id joins customers.id"]
}
```

### `POST /embed`

Returns the dense embedding for a piece of text.

Request body:

```json
{
	"text": "show total sales by month"
}
```

Response shape:

```json
{
	"embeddings": [0.01, -0.02, 0.13]
}
```

## n8n Workflow

The included [n8n-workflow.json](n8n-workflow.json) file implements the end-to-end orchestration. The workflow:

1. Accepts a user query through a webhook.
2. Calls `POST /embed` to get the query embedding.
3. Checks `public.query_cache` for a similar previously generated SQL statement.
4. If the cache hit is close enough, it reuses the cached SQL.
5. Otherwise, it calls `POST /retrieve` to get schema context.
6. Uses an LLM to generate a PostgreSQL `SELECT` query.
7. Validates the generated query with `EXPLAIN`.
8. Executes the SQL against PostgreSQL.
9. Stores the result in cache and returns the response.

The workflow depends on the API being reachable from n8n. In the export, the HTTP nodes point to `http://host.docker.internal:8000/embed` and `http://host.docker.internal:8000/retrieve`.

## How The Retrieval Layer Works

The retrieval engine in [app/services/retriever.py](app/services/retriever.py) combines three signals:

- table-level dense search with FAISS,
- graph-based join path discovery with NetworkX,
- column-level semantic search over precomputed table column embeddings.

This keeps the SQL generator focused on the smallest relevant slice of schema context instead of the full database structure.

## Customization

You can adapt the project to a different database or schema by replacing the files under `data/`:

- `schema.json` for table and column metadata
- `bi-encoder-metadata.json` for table-level dense search text
- `cross-encoder-metadata.json` for reranking metadata
- `graph.json` for table relationship and join information

If you change the schema, regenerate those artifacts so retrieval stays aligned with the target database.

## Notes

- The API currently runs the models on CPU.
- The project is optimized for PostgreSQL-backed SQL generation.
- The n8n workflow expects a `public.query_cache` table in PostgreSQL.
- Sample values can be omitted from `/retrieve` responses by setting `get_sample_values` to `false`.

## Example Usage

Once the API is running, you can test it with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
	-H "Content-Type: application/json" \
	-d "{\"query\": \"show total sales by month\", \"top_tables\": 5, \"top_columns\": 12, \"get_sample_values\": false}"
```
