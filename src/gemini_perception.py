# src/gemini_perception.py
import google.genai as genai
import os
import json
from PIL import Image
from dotenv import load_dotenv
from google.genai import types

class GeminiPerception:
    """
    Handles communication with the Google Gemini Pro Vision API,
    loading the API key from a .env file.
    """
    def __init__(self, model_name="gemini-robotics-er-1.5-preview"):
        self.model_name = model_name
        
        # --- THIS IS THE UPDATED SECTION ---
        # Load environment variables from a .env file in the project root
        load_dotenv() 
        
        # Configure the API key from the loaded environment variable
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Make sure it is set in your .env file.")
        
        self.client = genai.Client(api_key=api_key)
        print(f"GeminiPerception module initialized with model: {self.model_name}")

    def find_object_position(self, rgb_image, question):
        print("--- Sending Request to Gemini API ---")
        try:
            pil_img = Image.fromarray(rgb_image)
            

            image_response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    pil_img,
                question
            ],
            config = types.GenerateContentConfig(
      temperature=0.05,
      thinking_config=types.ThinkingConfig(thinking_budget=-1)
  )
        )
            
            cleaned_text = image_response.text.strip().replace("```json", "").replace("```", "").strip()
            
            print("--- Gemini Response Received ---")
            print(cleaned_text)
            
            return json.loads(cleaned_text)
            
        except Exception as e:
            print(f"ERROR: An error occurred during Gemini inference: {e}")
            return None