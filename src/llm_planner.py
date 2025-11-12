import ollama
import json

class LLMPlanner:
    """Handles communication with a text-based LLM for planning."""
    def __init__(self, model="llama3:8b"):
        self.model = model
        print(f"LLMPlanner initialized with model: {self.model}")

    def generate_plan(self, prompt):
        print("--- Sending Prompt to Planner LLM ---")
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                format="json"
            )
            content = response['message']['content']
            print("--- LLM Response Received ---")
            return json.loads(content)
        except Exception as e:
            print(f"An error occurred while communicating with the LLM: {e}")
            return None