import time
import logging
from app.services.llm import LLMService

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, name: str, system_prompt: str, llm_service: LLMService):
        self.name = name
        self.system_prompt = system_prompt
        self.llm_service = llm_service

    async def run(self, state: dict) -> dict:
        """
        Executes agent logic.
        Updates state and appends to the history log.
        """
        start_time = time.time()
        logger.info(f"Starting execution of Agent: {self.name}")
        
        try:
            # Subclasses override this method to perform specialized agent routines
            result_state = await self.process(state)
            
            elapsed_ms = (time.time() - start_time) * 1000.0
            step_record = {
                "agent": self.name,
                "status": "success",
                "latency_ms": round(elapsed_ms, 2),
                "timestamp": time.time()
            }
            # Copy updates
            result_state["history"] = result_state.get("history", []) + [step_record]
            return result_state
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000.0
            logger.error(f"Error in Agent {self.name}: {e}")
            step_record = {
                "agent": self.name,
                "status": "failed",
                "error": str(e),
                "latency_ms": round(elapsed_ms, 2),
                "timestamp": time.time()
            }
            state["history"] = state.get("history", []) + [step_record]
            state["security_blocked"] = True
            state["response"] = f"Internal processing error occurred at {self.name}."
            return state

    async def process(self, state: dict) -> dict:
        """Overridden by child classes."""
        return state
