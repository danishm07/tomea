import os
import logging
from openai import AsyncOpenAI
from typing import Optional

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model: str = "google/gemini-2.5-flash"):
        """
        Initializes the OpenRouter client.
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            # Fallback for local testing or if env var not set immediately
            logger.warning("OPENROUTER_API_KEY not found in environment.")
            
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.default_model = model

    async def generate(self, system_prompt: str, user_prompt: str, model: Optional[str] = None) -> str:
        """
        Generates a response from the LLM.
        """
        target_model = model or self.default_model
        try:
            response = await self.client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, # Low temp for code stability
                max_tokens=8000, # Allow for long implementations
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return f"# Error generating code: {e}"

    async def generate_fast(self, prompt: str) -> str:
        """Helper for small tasks (Router/Classifier/Summarization)"""
        # Using a cheaper/faster model for logic decisions
        return await self.generate(
            "You are a helpful logic assistant.", 
            prompt, 
            model="google/gemini-2.5-flash"
        )