import os
import base64
import json
from openai import OpenAI
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()

class VisionDataExtractor:
    def __init__(self):
        # We use the standard OpenAI client, but route it to OpenRouter
        # It automatically picks up the OPENROUTER_API_KEY from your environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        # Using the model you configured earlier
        self.model_name = "nvidia/nemotron-nano-12b-v2-vl:free"

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Reads an image file and converts it to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image(self, image_path: str, verbose: bool = True) -> Optional[str]:
        """
        Sends the image to the VLM. 
        Returns the description if useful, or None if it's a logo/junk.
        """
        if verbose: print(f"🔍 Analyzing image: {os.path.basename(image_path)}...")

        base64_image = self._encode_image_to_base64(image_path)

        # The strict system prompt to enforce JSON and act as a filter
        system_prompt = """
        You are an expert automotive data-extraction assistant. 
        Step 1: Analyze the image. Is it valuable technical data (flowcharts, engine schematics, diagnostic tables)? Or is it useless (company logos, decorative borders, generic photos)?
        Step 2: Return a strict JSON object with exactly two keys:
        - "is_useful": boolean (true if valuable, false if useless)
        - "description": string (If true, write a highly detailed summary of all text, flow, and data in the image. If false, leave empty).
        Output ONLY valid JSON. No markdown formatting blocks.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract the data from this image and return the JSON."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                # OpenRouter supports forcing JSON output for many models
                response_format={"type": "json_object"}, 
                temperature=0.1 # Keep it deterministic and factual
            )

            # Parse the JSON response
            raw_result = response.choices[0].message.content
            result_dict = json.loads(raw_result)

            if not result_dict.get("is_useful"):
                if verbose: print("   🚫 Image rejected (Logo/Decorative).")
                return None
            else:
                if verbose: print("   ✅ Image accepted! Data extracted.")
                return result_dict.get("description")

        except Exception as e:
            print(f"   ⚠️ Error processing image {image_path}: {str(e)}")
            return None

# --- Quick Test Block ---
if __name__ == "__main__":
    # Ensure you have set your API key in the terminal before running:
    # setx OPENROUTER_API_KEY "your-key-here" (and restart terminal)
    
    extractor = VisionDataExtractor()
    
    # Point this to an actual image extracted by your Docling/PyMuPDF script
    test_image = r"/workspaces/Multimodal_rag/images.png"
    
    if os.path.exists(test_image):
        description = extractor.analyze_image(test_image)
        if description:
            print(f"\nExtracted Text to Embed:\n{description}")
    else:
        print("Please provide a valid image path to test.")