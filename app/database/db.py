import sqlite3
import json
import os
import uuid
import time
import re
import numpy as np
import logging
from app.config import settings

logger = logging.getLogger(__name__)

# Security Clearance Levels
ROLE_HIERARCHY = {
    "Executive": ["Executive", "Manager", "Employee", "Public"],
    "Manager": ["Manager", "Employee", "Public"],
    "Employee": ["Employee", "Public"],
    "Public": ["Public"]
}

def get_allowed_roles(user_role: str) -> list[str]:
    """Returns list of required_roles a user role is allowed to view."""
    return ROLE_HIERARCHY.get(user_role, ["Public"])

def mask_text_rbac(text: str, user_role: str) -> str:
    """
    Dynamically masks text blocks that require higher clearance than the user's role.
    Example text: "HQ is in Hyderabad. [Executive: CEO salary is $1M] Support is support@secure.com"
    If user is 'Employee', it becomes: "HQ is in Hyderabad. [ACCESS RESTRICTED: EXECUTIVE Clearance Required] Support is support@secure.com"
    """
    pattern = re.compile(r"\[(\w+):\s*(.*?)\]")
    allowed_roles = get_allowed_roles(user_role)
    
    def replacer(match):
        required_role = match.group(1)
        content = match.group(2)
        # If the required role clearance is allowed for the user, return clean content
        if required_role in allowed_roles:
            return content
        # Otherwise, redact/mask it securely
        return f"[ACCESS RESTRICTED: {required_role.upper()} Clearance Required]"
        
    return pattern.sub(replacer, text)

class DatabaseManager:
    def __init__(self, db_path: str = settings.DATABASE_PATH):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes database schemas for documents and security audit logs."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 1. Documents table with security clearance mapping
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                required_role TEXT NOT NULL,
                metadata TEXT
            )
        """)

        # 2. Audit logs for tracing agent steps & security reviews
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                user_role TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                trace_steps TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("SQLite database initialized successfully.")

    def add_document(self, text: str, embedding: list[float], required_role: str = "Public", doc_metadata: dict = None) -> str:
        """Adds a new document chunk to the database."""
        doc_id = str(uuid.uuid4())
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO documents (id, text, embedding, required_role, metadata) VALUES (?, ?, ?, ?, ?)",
            (
                doc_id,
                text,
                json.dumps(embedding),
                required_role,
                json.dumps(doc_metadata or {})
            )
        )
        conn.commit()
        conn.close()
        return doc_id

    def retrieve_context_rbac(self, query_embedding: list[float], user_role: str, limit: int = 3, threshold: float = settings.SIMILARITY_THRESHOLD) -> list[dict]:
        """
        Retrieves context chunks matching the query embedding, restricted by the user's role.
        Enforces strict RBAC filtering at the SQL level BEFORE executing similarity checks.
        Applies dynamic entity-level field masking for mixed-role documents.
        """
        allowed_roles = get_allowed_roles(user_role)
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # SQL-level filter: Select only documents whose required_role is allowed for the user
        placeholders = ",".join("?" for _ in allowed_roles)
        query = f"SELECT id, text, required_role, embedding, metadata FROM documents WHERE required_role IN ({placeholders})"
        cursor.execute(query, allowed_roles)
        rows = cursor.fetchall()
        conn.close()

        results = []
        q_vec = np.array(query_embedding)
        q_norm = np.linalg.norm(q_vec)
        
        if q_norm == 0.0:
            return []

        for row in rows:
            doc_emb = np.array(json.loads(row["embedding"]))
            doc_norm = np.linalg.norm(doc_emb)
            if doc_norm == 0.0:
                continue
                
            similarity = float(np.dot(q_vec, doc_emb) / (q_norm * doc_norm))
            if similarity >= threshold:
                # Apply dynamic, on-the-fly entity-level text masking
                masked_text = mask_text_rbac(row["text"], user_role)
                
                results.append({
                    "id": row["id"],
                    "text": masked_text,
                    "required_role": row["required_role"],
                    "similarity": similarity,
                    "metadata": json.loads(row["metadata"] or "{}")
                })
        
        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def add_audit_log(self, task_id: str, user_role: str, prompt: str, response: str, trace_steps: list[dict]) -> str:
        """Saves a task run to audit log for history/compliance tracing."""
        log_id = str(uuid.uuid4())
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO audit_logs (id, task_id, user_role, prompt, response, trace_steps, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                log_id,
                task_id,
                user_role,
                prompt,
                response,
                json.dumps(trace_steps),
                time.time()
            )
        )
        conn.commit()
        conn.close()
        return log_id

    def get_audit_logs(self, limit: int = 20) -> list[dict]:
        """Returns all audit logs sorted by timestamp descending."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, task_id, user_role, prompt, response, trace_steps, timestamp FROM audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        logs = []
        for row in rows:
            logs.append({
                "id": row["id"],
                "task_id": row["task_id"],
                "user_role": row["user_role"],
                "prompt": row["prompt"],
                "response": row["response"],
                "trace_steps": json.loads(row["trace_steps"]),
                "timestamp": row["timestamp"]
            })
        return logs

    def clear_documents(self):
        """Clears all documents in index."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents")
        conn.commit()
        conn.close()
