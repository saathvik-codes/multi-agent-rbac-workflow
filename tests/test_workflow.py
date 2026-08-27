import pytest
from fastapi.testclient import TestClient
from app.main import app, get_db, get_embedder, get_workflow
from app.database.db import DatabaseManager
from app.services.embedding import EmbeddingService
from app.agents.workflow import EnterpriseWorkflowEngine
import unittest.mock as mock

@pytest.fixture
def mock_dependencies(tmp_path):
    # Dynamic temp DB
    db_file = tmp_path / "test_workflow.db"
    db = DatabaseManager(str(db_file))
    
    embedder = EmbeddingService()
    from app.services.llm import LLMService
    llm = LLMService()
    
    # Mock LLM generation to be completely deterministic in tests
    async def mock_generate(system_prompt: str, user_prompt: str, *args, **kwargs):
        sys_lower = system_prompt.lower()
        user_lower = user_prompt.lower()
        if "router" in sys_lower:
            return "CLEAN"
        if "analyst" in sys_lower:
            if "no retrieved context" in user_lower or "no source document" in user_lower:
                return "Based on secure corporate records, no relevant facts were found matching your security clearance level."
            if "[system feedback]" in user_lower:
                # Simulated self-correction: Analyst rewrites output to remove the phone number!
                return "Based on secure corporate records, the request is resolved. The sensitive phone number has been omitted to comply with policy."
            return f"Based on secure corporate records, here is the context: {user_prompt}"
        if "guardrail" in sys_lower:
            # Replicate PII email redaction logic mock
            import re
            output = user_prompt
            output = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', output)
            return output
        return "Mock response"
        
    llm.generate = mock.AsyncMock(side_effect=mock_generate)
    workflow = EnterpriseWorkflowEngine(db, embedder, llm)
    
    # Overwrite dependencies
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_workflow] = lambda: workflow
    
    yield db, embedder
    
    app.dependency_overrides.clear()
    db.clear_documents()

def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_document_indexing(mock_dependencies):
    db, embedder = mock_dependencies
    client = TestClient(app)
    
    response = client.post("/api/document", json={
        "text": "Executive confidential roadmap file",
        "required_role": "Executive",
        "metadata": {"title": "Roadmap"}
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["required_role"] == "Executive"
    
    # Check that it exists in SQLite
    docs = client.get("/api/documents").json()
    assert len(docs) == 1
    assert docs[0]["required_role"] == "Executive"
    assert docs[0]["text"] == "Executive confidential roadmap file"

def test_workflow_end_to_end_public_blocked(mock_dependencies):
    db, embedder = mock_dependencies
    client = TestClient(app)
    
    # Index executive file
    client.post("/api/document", json={
        "text": "Executive financial data containing secret email secure@corp.com",
        "required_role": "Executive"
    })
    
    # Run workflow query as PUBLIC (Clearance should prevent document retrieval)
    response = client.post("/api/task", json={
        "prompt": "Show executive financial data",
        "user_role": "Public"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["security_blocked"] is False
    assert len(data["context"]) == 0
    resp_lower = data["response"].lower()
    assert any(x in resp_lower for x in ["no relevant facts", "no retrieved context", "not grant access"])
    
    # Audit log check
    logs = client.get("/api/logs").json()
    assert len(logs) == 1
    assert logs[0]["user_role"] == "Public"

def test_workflow_end_to_end_executive_allowed(mock_dependencies):
    db, embedder = mock_dependencies
    client = TestClient(app)
    
    # Index executive file
    client.post("/api/document", json={
        "text": "Executive financial data containing secret email secure@corp.com",
        "required_role": "Executive"
    })
    
    # Run workflow query as EXECUTIVE (Clearance should allow, output guardrail should redact email)
    response = client.post("/api/task", json={
        "prompt": "Show executive financial data",
        "user_role": "Executive"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["security_blocked"] is False
    assert len(data["context"]) == 1
    assert data["context"][0]["required_role"] == "Executive"
    
    # Output should contain context but HAVE email redacted by GuardrailAgent!
    assert "[REDACTED_EMAIL]" in data["response"]
    assert "secure@corp.com" not in data["response"]
    assert "PII: Emails redacted." in data["security_warnings"]

def test_workflow_self_correction_loop(mock_dependencies):
    db, embedder = mock_dependencies
    client = TestClient(app)
    
    # Index executive file with sensitive phone number (policy violation)
    client.post("/api/document", json={
        "text": "Executive team database contacts list. Chief number is 555-019-2828.",
        "required_role": "Executive"
    })
    
    # Query as Executive.
    # The first Analyst generation will include the phone number: 555-019-2828.
    # The first Guardrail check will catch it, reject it, and request correction.
    # The second Analyst generation will check system feedback and return the clean omitted version.
    response = client.post("/api/task", json={
        "prompt": "List team contacts",
        "user_role": "Executive"
    })
    
    assert response.status_code == 200
    data = response.json()
    
    # 1. State machine succeeded without hard blocks
    assert data["security_blocked"] is False
    
    # 2. Output contains the self-corrected message (no phone number)
    assert "omitted to comply with policy" in data["response"]
    assert "555-019-2828" not in data["response"]
    
    # 3. History log lists multiple analyst and guardrail steps (proving loop was run!)
    history = data["history"]
    analyst_steps = [h for h in history if h.get("agent") == "AnalystAgent"]
    guardrail_steps = [h for h in history if h.get("agent") == "GuardrailAgent"]
    
    assert len(analyst_steps) == 2
    assert len(guardrail_steps) == 2
