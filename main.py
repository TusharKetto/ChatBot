from fastapi import FastAPI
from pydantic import BaseModel
from LLMQuery import query_llm
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Ketto Knowledge Base Query API",
    description="API to query the Ketto Knowledge Base using a Gen AI + BigQuery pipeline",
    version="1.0.0"
)

# Allow frontend origins (Live Server or static HTML)
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],       # Use ["*"] for dev/testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    user_query: str

class QueryResponse(BaseModel):
    response: str

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        response_text = query_llm(request.user_query)
        return QueryResponse(response=response_text)
    except Exception as e:
        return QueryResponse(response=f"Error: {str(e)}")
