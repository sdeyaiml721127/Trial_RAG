# p8_image_processor.py
"""
Handles Image Processing: Visual Description.
"""
import base64
import os
import httpx 
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage
import p1_config as config

# Initialize httpx client
client = httpx.Client(verify=False)

class ImageProcessor:
    def __init__(self):
        # Initialize Vision Model
        self.vision_llm = ChatOpenAI(
            base_url="https://genailab.tcs.in",
            model=config.VISION_MODEL_NAME, 
            api_key=config.MAAS_API_KEY, 
            temperature=0.1,
            http_client=client 
        )
        print(f"ImageProcessor initialized with model: {config.VISION_MODEL_NAME}")

    def _encode_image(self, image_path: str) -> str:
        """Encodes an image file to base64 string."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image(self, image_path: str) -> str:
        """
        Generates a detailed description of the image.
        """
        try:
            print(f"   -> Analyzing image: {image_path}...")
            base64_image = self._encode_image(image_path)
            
            # Construct the prompt for the Vision model
            messages = [
                {
                    "type": "text", 
                    "text": (
                        "Analyze this image and provide a highly detailed, objective description. "
                        "List all visible objects, people, actions, and the setting/environment. "
                        "Do not hallucinate details." # UPDATED PROMPT
                    )
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
            
            # Invoke Vision LLM
            response = self.vision_llm.invoke([HumanMessage(content=messages)]) 
            return response.content

        except Exception as e:
            return f"[Image Analysis Failed]: {str(e)}"