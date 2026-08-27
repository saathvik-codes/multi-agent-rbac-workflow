import uuid
import logging
from app.database.db import DatabaseManager
from app.services.embedding import EmbeddingService
from app.services.llm import LLMService
from app.agents.router import GatewayRouterAgent
from app.agents.retrieval import SecureRetrievalAgent
from app.agents.analyst import AnalystAgent
from app.agents.guardrail import GuardrailAgent

logger = logging.getLogger(__name__)

class EnterpriseWorkflowEngine:
    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService, llm_service: LLMService):
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        
        # Instantiate specialized agents
        self.router_agent = GatewayRouterAgent(self.llm_service)
        self.retrieval_agent = SecureRetrievalAgent(self.llm_service, self.db_manager, self.embedding_service)
        self.analyst_agent = AnalystAgent(self.llm_service)
        self.guardrail_agent = GuardrailAgent(self.llm_service)

    async def execute(self, prompt: str, user_role: str) -> dict:
        """
        Coordinates the execution trace path of agents in a state machine pipeline.
        Supports cyclic self-correcting loops for security violations.
        """
        task_id = str(uuid.uuid4())
        
        # Initialize workflow state
        state = {
            "task_id": task_id,
            "prompt": prompt,
            "user_role": user_role,
            "history": [],
            "context": [],
            "response": "",
            "security_blocked": False,
            "security_warnings": [],
            "guardrail_attempts": 0,
            "needs_correction": False,
            "feedback_message": ""
        }

        logger.info(f"Triggering execution trace for Task {task_id} (Role: {user_role})...")

        # Step 1: Input injection guardrail scan
        state = await self.router_agent.run(state)
        if state.get("security_blocked", False):
            logger.warning(f"Workflow halted early due to security block at GatewayRouterAgent.")
            self.db_manager.add_audit_log(
                task_id=task_id,
                user_role=user_role,
                prompt=prompt,
                response=state["response"],
                trace_steps=state["history"]
            )
            return state

        # Step 2: RBAC-restricted vector retrieval (RAG)
        state = await self.retrieval_agent.run(state)

        # Steps 3 & 4: Cyclic Analyst & Guardrail Loop
        max_attempts = 2
        
        while state["guardrail_attempts"] < max_attempts:
            if state.get("security_blocked", False):
                break
                
            # Run analyst to generate draft
            state = await self.analyst_agent.run(state)
            # Scan output using guardrail
            state = await self.guardrail_agent.run(state)
            
            # If no corrections are required, break loop
            if not state.get("needs_correction", False):
                break
                
            logger.info(
                f"Self-correction triggered (Attempt {state['guardrail_attempts']}/{max_attempts}). "
                f"Feedback: {state['feedback_message']}"
            )
            
            # Adjust the prompt state to include feedback instructions
            original_prompt = state.get("prompt", "")
            feedback = state.get("feedback_message", "")
            
            clean_prompt = original_prompt.split("\n\n[SYSTEM FEEDBACK]:")[0]
            state["prompt"] = f"{clean_prompt}\n\n[SYSTEM FEEDBACK]: {feedback}"
            state["needs_correction"] = False  # Reset flag for next run

        # Audit logging: Save run details to SQL DB
        self.db_manager.add_audit_log(
            task_id=task_id,
            user_role=user_role,
            prompt=prompt,
            response=state["response"],
            trace_steps=state["history"]
        )

        logger.info(f"Workflow execution completed for Task {task_id}.")
        return state
