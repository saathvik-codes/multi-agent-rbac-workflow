import re
import logging
from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
API_KEY_PATTERN = re.compile(r"(?i)\b(api_key|secret|password|passwd|token)\b\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]")

class GuardrailAgent(BaseAgent):
    def __init__(self, llm_service):
        system_prompt = (
            "You are the Output Guardrail Agent. Your job is to check the generated response for "
            "policy violations, toxic statements, or data leaks. If clean, reply with exactly the input text."
        )
        super().__init__("GuardrailAgent", system_prompt, llm_service)

    async def process(self, state: dict) -> dict:
        if state.get("security_blocked", False):
            return state

        response_text = state.get("response", "")
        if not response_text:
            return state

        # Track attempts to prevent infinite loops
        attempts = state.get("guardrail_attempts", 0) + 1
        state["guardrail_attempts"] = attempts

        warnings = []
        needs_correction = False
        feedback_msg = ""

        # 1. Check for API keys or Secrets (High-risk policy violation)
        secrets = API_KEY_PATTERN.findall(response_text)
        if secrets:
            logger.warning("Guardrail Agent: API key or secret detected in generated draft.")
            needs_correction = True
            feedback_msg += "Policy Violation: The generated response contains hardcoded passwords or API tokens. Please rewrite the response without including these secrets. "

        # 2. Check for Phone numbers (Sensitive PII policy violation)
        phones = PHONE_PATTERN.findall(response_text)
        if phones:
            logger.warning("Guardrail Agent: Phone number PII detected in generated draft.")
            needs_correction = True
            feedback_msg += "Policy Violation: The generated response exposes sensitive phone numbers. Please rewrite the response without containing phone numbers. "

        # If violations are found and we haven't exceeded maximum retries (e.g. 2), trigger correction loop
        if needs_correction and attempts <= 2:
            logger.info(f"Guardrail Agent: Requesting self-correction loop (Attempt {attempts}).")
            state["needs_correction"] = True
            state["feedback_message"] = feedback_msg.strip()
            return state

        # 3. Fallback: If we exceed retries OR for other minor PII (like emails), apply direct redaction
        redacted_text = response_text
        emails = EMAIL_PATTERN.findall(redacted_text)
        if emails:
            logger.warning("Guardrail Agent: Redacting emails from final output.")
            redacted_text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted_text)
            warnings.append("PII: Emails redacted.")

        if secrets:
            logger.warning("Guardrail Agent: Redacting credentials as safety fallback.")
            redacted_text = API_KEY_PATTERN.sub(r"\1: [REDACTED_SECRET]", redacted_text)
            warnings.append("Security: Secrets redacted.")

        if phones:
            logger.warning("Guardrail Agent: Redacting phone numbers as safety fallback.")
            redacted_text = PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted_text)
            warnings.append("PII: Phone numbers redacted.")

        state["response"] = redacted_text
        state["needs_correction"] = False
        if warnings:
            state["security_warnings"] = state.get("security_warnings", []) + warnings

        logger.info("Guardrail Agent completed response scanning.")
        return state
