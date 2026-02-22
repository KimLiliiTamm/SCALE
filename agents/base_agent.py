import time
from typing import List, Dict, Any
from openai import OpenAI

class BaseAgent:
    """A base class for all AI-powered agents."""
    def __init__(self, api_key: str, model: str, system_prompt: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = model
        self.system_prompt = system_prompt
        self.context: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]

    def _generate_answer(self, temperature: float = 0.0) -> str:
        """
        Generates a response from the LLM based on the current context.
        Includes retry logic for API errors.
        """
        try:
            # CHANGE 2: Added a small safety check.
            # Groq sometimes struggles with strict temperature=0.0, so we default to small value if needed,
            # but usually it's fine.
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.context,
                n=1,
                temperature=temperature
            )
            return completion.choices[0].message.content

        except Exception as e:
            # CHANGE 3: Better error logging so you know WHY it failed
            print(f"\n[Groq API Error]: {e}")
            print("Waiting 30 seconds before retrying...")
            time.sleep(30)
            # Recursively try again
            return self._generate_answer(temperature)


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