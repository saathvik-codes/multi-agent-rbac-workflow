import logging
from app.agents.base import BaseAgent
from app.database.db import DatabaseManager
from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

class SecureRetrievalAgent(BaseAgent):
    def __init__(self, llm_service, db_manager: DatabaseManager, embedding_service: EmbeddingService):
        system_prompt = (
            "You are the Secure Retrieval Agent. Your job is to extract relevant facts from the database "
            "while strictly adhering to safety credentials and RBAC clearances."
        )
        super().__init__("SecureRetrievalAgent", system_prompt, llm_service)
        self.db_manager = db_manager
        self.embedding_service = embedding_service

    async def process(self, state: dict) -> dict:
        # Check if previous agent blocked execution
        if state.get("security_blocked", False):
            return state

        prompt = state.get("prompt", "")
        user_role = state.get("user_role", "Public")
        
        logger.info(f"Generating embedding query for RBAC RAG (Role: {user_role})...")
        try:
            # 1. Get vector embedding of user query
            query_emb = await self.embedding_service.get_embedding(prompt)
            
            # 2. Query DB with strict RBAC boundary checks
            limit = 3
            contexts = self.db_manager.retrieve_context_rbac(
                query_embedding=query_emb,
                user_role=user_role,
                limit=limit
            )
            
            # 3. Save to state memory
            state["context"] = contexts
            logger.info(f"Retrieved {len(contexts)} secure context chunks for user role '{user_role}'.")
            
        except Exception as e:
            logger.error(f"Error during RAG retrieval: {e}")
            state["context"] = []
            state["security_warnings"] = state.get("security_warnings", []) + [f"Retrieval failed: {e}"]

        return state
