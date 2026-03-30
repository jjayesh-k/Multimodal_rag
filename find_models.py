import urllib.request
import json

print("Fetching live models from OpenRouter...\n")

url = "https://openrouter.ai/api/v1/models"
req = urllib.request.Request(url)

try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    
    # Lists to store our findings
    free_vision_models = []
    free_text_models = []
    
    for m in data.get("data", []):
        # 1. Check if the model is completely free
        pricing = m.get("pricing", {})
        is_free = str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0"
        
        if is_free:
            name = m["id"].lower()
            modality = m.get("architecture", {}).get("modality", "")
            
            # 2. Check for Vision capability
            if "image" in modality.lower() or "vision" in name:
                free_vision_models.append(m["id"])
            else:
                # 3. Otherwise, categorize as Text/Language
                free_text_models.append(m["id"])

    print("🟢 AVAILABLE FREE VISION MODELS (Use for Image Summaries):")
    print("-" * 60)
    for model_id in free_vision_models:
        print(f"  - {model_id}")
    if not free_vision_models: print("  None found.")

    print("\n🔵 AVAILABLE FREE LANGUAGE MODELS (Use for main.py GENERATION_MODEL):")
    print("-" * 60)
    for model_id in free_text_models:
        print(f"  - {model_id}")
    if not free_text_models: print("  None found.")
        
except Exception as e:
    print(f"Failed to fetch models: {e}")