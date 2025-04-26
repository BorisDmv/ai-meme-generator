from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import requests
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# --- Settings ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HUGGINGFACE_MODEL_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# --- Load BLIP locally for image captioning ---
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    token=HUGGINGFACE_API_TOKEN
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    token=HUGGINGFACE_API_TOKEN
)

# --- Functions ---
def generate_caption(image_path):
    print("Thinking... Generating caption for the image...")
    image = Image.open(image_path).convert('RGB')
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    print("Caption generated:", caption)
    return caption

def make_it_funny(caption):
    instruction = (
        f"Write one single short sentence as a funny meme caption for this: {caption}. "
        "Do not write more than one sentence. End with a period."
    )
    
    # Wrap instruction properly for Instruct models
    prompt = f"<s>[INST] {instruction} [/INST]"

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 25,
            "do_sample": True,
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 50,
            "stop": ["</s>"]
        }
    }

    # Perform the API request
    response = requests.post(HUGGINGFACE_MODEL_URL, headers=headers, json=payload)
    
    # Debugging the response
    print("Response Status Code:", response.status_code)
    print("Response Text:", response.text)
    
    if response.status_code != 200:
        print("Error from Huggingface API:", response.text)
        return "Error generating meme."

    result = response.json()

    full_text = result[0]['generated_text'] if isinstance(result, list) else result.get('generated_text', '')

    # --- Clean the output ---
    # Remove the prompt part
    if "[/INST]" in full_text:
        meme = full_text.split("[/INST]")[-1].strip()
    else:
        meme = full_text.strip()

    return meme



# --- Main ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_meme_text.py path_to_image")
    else:
        print("Starting meme generation process...")
        caption = generate_caption(sys.argv[1])
        funny = make_it_funny(caption)
        print("Final Funny Meme Text:", funny)