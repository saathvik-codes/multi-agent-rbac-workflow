import pytest
from app.database.db import DatabaseManager, ROLE_HIERARCHY

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_rbac.db"
    db = DatabaseManager(str(db_file))
    yield db
    # Cleanup
    db.clear_documents()

def test_role_hierarchy_permissions():
    assert "Executive" in ROLE_HIERARCHY["Executive"]
    assert "Public" in ROLE_HIERARCHY["Public"]
    assert "Executive" not in ROLE_HIERARCHY["Public"]
    assert "Manager" not in ROLE_HIERARCHY["Employee"]
    assert "Employee" in ROLE_HIERARCHY["Manager"]

def test_rbac_context_filtering(temp_db):
    # Dummy embedding dimensions
    emb_doc = [0.5] * 768
    
    # Insert multiple documents with different required roles
    temp_db.add_document("Public info text", emb_doc, "Public")
    temp_db.add_document("Employee info text", emb_doc, "Employee")
    temp_db.add_document("Manager info text", emb_doc, "Manager")
    temp_db.add_document("Executive confidential data", emb_doc, "Executive")

    query_emb = [0.5] * 768  # Perfect match

    # 1. Query as Public
    public_results = temp_db.retrieve_context_rbac(query_emb, "Public", threshold=0.90)
    assert len(public_results) == 1
    assert public_results[0]["text"] == "Public info text"

    # 2. Query as Employee
    employee_results = temp_db.retrieve_context_rbac(query_emb, "Employee", threshold=0.90)
    assert len(employee_results) == 2
    retrieved_texts = [r["text"] for r in employee_results]
    assert "Public info text" in retrieved_texts
    assert "Employee info text" in retrieved_texts
    assert "Executive confidential data" not in retrieved_texts

    # 3. Query as Executive
    exec_results = temp_db.retrieve_context_rbac(query_emb, "Executive", limit=10, threshold=0.90)
    assert len(exec_results) == 4
    all_texts = [r["text"] for r in exec_results]
    assert "Executive confidential data" in all_texts

def test_dynamic_entity_masking(temp_db):
    from app.database.db import mask_text_rbac
    
    # 1. Test raw masking function
    raw_text = "HQ is in Hyderabad. [Executive: CEO salary is $1M] Contact us at support@secure.com"
    
    # Employee query (should mask Executive section)
    emp_masked = mask_text_rbac(raw_text, "Employee")
    assert "CEO salary is $1M" not in emp_masked
    assert "[ACCESS RESTRICTED: EXECUTIVE Clearance Required]" in emp_masked
    
    # Executive query (should show clean text)
    exec_clean = mask_text_rbac(raw_text, "Executive")
    assert "CEO salary is $1M" in exec_clean
    assert "[ACCESS RESTRICTED" not in exec_clean
    
    # 2. Test database retrieval integration
    emb_doc = [0.5] * 768
    # Document itself is Employee clearance, but contains an Executive section!
    temp_db.add_document("We have employee lounges. [Executive: Board meeting room key is 9982]", emb_doc, "Employee")
    
    query_emb = [0.5] * 768
    
    # Query as Employee
    emp_res = temp_db.retrieve_context_rbac(query_emb, "Employee", threshold=0.90)
    assert len(emp_res) == 1
    assert "Board meeting room key is 9982" not in emp_res[0]["text"]
    assert "[ACCESS RESTRICTED: EXECUTIVE Clearance Required]" in emp_res[0]["text"]
    
    # Query as Executive
    exec_res = temp_db.retrieve_context_rbac(query_emb, "Executive", threshold=0.90)
    assert len(exec_res) == 1
    assert "Board meeting room key is 9982" in exec_res[0]["text"]
    assert "[ACCESS RESTRICTED" not in exec_res[0]["text"]
