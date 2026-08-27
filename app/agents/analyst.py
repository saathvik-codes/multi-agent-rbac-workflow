import logging
from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

class AnalystAgent(BaseAgent):
    def __init__(self, llm_service):
        system_prompt = (
            "You are the Enterprise Analyst Agent.\n"
            "Your objective is to answer the user's query utilizing ONLY the provided secure retrieved context.\n"
            "Ground your response strictly in the facts provided. If the retrieved context is empty, "
            "inform the user that no relevant facts were found matching their security clearance level.\n"
            "Ensure a professional corporate tone. Do not make up facts."
        )
        super().__init__("AnalystAgent", system_prompt, llm_service)

    async def process(self, state: dict) -> dict:
        # Check if previous agent blocked execution
        if state.get("security_blocked", False):
            return state

        prompt = state.get("prompt", "")
        contexts = state.get("context", [])
        
        # Format the retrieved contexts cleanly
        formatted_context = ""
        if contexts:
            for idx, ctx in enumerate(contexts):
                formatted_context += f"\n--- Source Document {idx+1} (Required Security Role: {ctx['required_role']}) ---\n"
                formatted_context += f"{ctx['text']}\n"
        else:
            formatted_context = "No retrieved context available."

        user_input_to_llm = (
            f"User Query:\n{prompt}\n\n"
            f"Retrieved Context:\n{formatted_context}"
        )

        try:
            draft_response = await self.llm_service.generate(
                system_prompt=self.system_prompt,
                user_prompt=user_input_to_llm,
                temperature=0.3,
                max_tokens=600
            )
            state["response"] = draft_response
            logger.info("Analyst Agent successfully formulated draft response.")
        except Exception as e:
            logger.error(f"Error in AnalystAgent text generation: {e}")
            state["response"] = "Error: Failed to analyze retrieved context."
            state["security_blocked"] = True

        return state
