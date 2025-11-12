import ollama
import json
from PIL import Image
import io

class VLMPception:
    """Handles communication with a Vision-Language Model for perception."""
    def __init__(self, model="qwen3-vl:8b"):
        self.model = model
        print(f"VLMPception initialized with model: {self.model}")

    def find_object_position(self, rgb_image, question):
        print("--- Preparing Image for VLM ---")
        try:
            pil_img = Image.fromarray(rgb_image)
            with io.BytesIO() as buffer:
                pil_img.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
        except Exception as e:
            print(f"ERROR: Failed during image preparation: {e}")
            return None

        print("--- Sending Request to VLM ---")
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': question,
                    'images': [image_bytes]
                }],
                format="json"
            )
            content = response['message']['content']
            print("--- VLM Response Received ---")
            return json.loads(content)
        except Exception as e:
            print(f"ERROR: An error occurred during VLM inference: {e}")
            return None