import time
import httpx
from typing import List, Dict, Any
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError

class BaseAgent:
    """A base class for all AI-powered agents."""
    def __init__(self, api_key: str, model: str, system_prompt: str, base_url: str = "https://api.groq.com/openai/v1", max_retries: int = 3):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.system_prompt = system_prompt
        self.context: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.max_retries = max_retries

    def _generate_answer(self, temperature: float = 0.0) -> str:
        """
        Generates a response from the LLM based on the current context.
        Retries up to self.max_retries times on transient errors only.
        Non-retryable errors (4xx) are raised immediately.
        """
        for attempt in range(self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.context,
                    n=1,
                    temperature=temperature
                )
                return completion.choices[0].message.content

            except (RateLimitError, APIConnectionError, httpx.ConnectError) as e:
                print(f"\n[Transient LLM Error, attempt {attempt + 1}/{self.max_retries + 1}]: {e}")
                if attempt < self.max_retries:
                    print("Waiting 30 seconds before retrying...")
                    time.sleep(30)
                else:
                    print("Max retries exhausted.")
                    raise

            except APIStatusError as e:
                if e.status_code >= 500:
                    print(f"\n[Server Error {e.status_code}, attempt {attempt + 1}/{self.max_retries + 1}]: {e}")
                    if attempt < self.max_retries:
                        print("Waiting 30 seconds before retrying...")
                        time.sleep(30)
                    else:
                        print("Max retries exhausted.")
                        raise
                else:
                    print(f"\n[Non-retryable LLM Error {e.status_code}]: {e}")
                    raise


    def add_user_message(self, content: str):
        """Adds a user message to the agent's context."""
        self.context.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """Adds an assistant message to the agent's context."""
        self.context.append({"role": "assistant", "content": content})
        
    def get_last_response(self) -> str:
        """Returns the last assistant response from the context."""
        for message in reversed(self.context):
            if message["role"] == "assistant":
                return message["content"]
        return ""

    def reset_context(self):
        """Resets the conversation context to just the system prompt."""
        self.context = [{"role": "system", "content": self.system_prompt}]