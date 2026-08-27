import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

from app.config import settings
from app.models import TaskRequest, TaskResponse, DocumentUploadRequest
from app.database.db import DatabaseManager
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.agents.workflow import EnterpriseWorkflowEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("workflow_server")

db_manager = None
embedding_service = None
llm_service = None
workflow_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_manager, embedding_service, llm_service, workflow_engine
    logger.info("Initializing workflow services...")
    db_manager = DatabaseManager()
    embedding_service = EmbeddingService()
    llm_service = LLMService()
    workflow_engine = EnterpriseWorkflowEngine(db_manager, embedding_service, llm_service)
    
    yield
    
    logger.info("Tearing down services...")
    await embedding_service.close()
    await llm_service.close()

app = FastAPI(
    title="Secure Multi-Agent RBAC Workflow",
    description="Enterprise-ready multi-agent coordinate engine with Role-Based Access Control and Input/Output Guardrails",
    version="1.0.0",
    lifespan=lifespan
)

# Dependency Injection helpers
def get_db():
    return db_manager

def get_embedder():
    return embedding_service

def get_workflow():
    return workflow_engine

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the front-end dashboard."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Dashboard template index.html not found.")
    
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/api/task", response_model=TaskResponse)
async def run_workflow(request: TaskRequest, engine: EnterpriseWorkflowEngine = Depends(get_workflow)):
    """Triggers the multi-agent orchestration execution trace."""
    try:
        result = await engine.execute(request.prompt, request.user_role)
        return result
    except Exception as e:
        logger.error(f"Error executing agent workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Agent workflow execution failed: {e}")

@app.post("/api/document")
async def add_document(
    request: DocumentUploadRequest,
    db: DatabaseManager = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder)
):
    """Embeds and saves a document chunk restricted by access clearance level."""
    try:
        # Get embedding vector (runs local Ollama/mock fallback)
        embedding = await embedder.get_embedding(request.text)
        
        # Add to SQLite
        doc_id = db.add_document(
            text=request.text,
            embedding=embedding,
            required_role=request.required_role,
            doc_metadata=request.metadata
        )
        return {"status": "success", "document_id": doc_id, "required_role": request.required_role}
    except Exception as e:
        logger.error(f"Failed to index document chunk: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to index document: {e}")

@app.get("/api/documents")
async def get_documents(db: DatabaseManager = Depends(get_db)):
    """Returns all stored documents in index."""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, required_role, metadata FROM documents")
        rows = cursor.fetchall()
        conn.close()
        
        docs = []
        for r in rows:
            import json
            docs.append({
                "id": r["id"],
                "text": r["text"],
                "required_role": r["required_role"],
                "metadata": json.loads(r["metadata"] or "{}")
            })
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs")
async def get_logs(db: DatabaseManager = Depends(get_db)):
    """Retrieves all security audit logs."""
    return db.get_audit_logs(limit=25)

@app.post("/api/clear")
async def clear_db(db: DatabaseManager = Depends(get_db)):
    """Resets the vector index and document database."""
    db.clear_documents()
    return {"message": "Database documents cleared successfully"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_connected": os.path.exists(settings.DATABASE_PATH),
        "gemini_api_key_configured": bool(settings.GEMINI_API_KEY)
    }
