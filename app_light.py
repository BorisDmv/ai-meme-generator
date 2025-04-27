from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import requests
from dotenv import load_dotenv
from io import BytesIO
import random
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
def generate_caption(image):
    """Generate a caption for a PIL Image object."""
    print("Thinking... Generating caption for the image...")
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    print("Caption generated:", caption)
    return caption

def random_instruction(caption):
    """Generate a slightly varied instruction to add randomness."""
    templates = [
        f"Write a funny one-line meme about: {caption}",
        f"Make a hilarious meme caption for: {caption}",
        f"Create a short meme text describing: {caption}",
        f"Come up with a meme for: {caption}",
        f"Write a clever meme line for: {caption}",
        f"Invent a funny short meme about: {caption}",
    ]
    return random.choice(templates)

def make_it_funny(caption):
    """Send the caption to HuggingFace model to make it funny."""
    instruction = random_instruction(caption)
    prompt = f"<s>[INST] {instruction} [/INST]"

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 50,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "stop": ["</s>"]
        }
    }

    response = requests.post(HUGGINGFACE_MODEL_URL, headers=headers, json=payload)
    
    print("Response Status Code:", response.status_code)
    print("Response Text:", response.text)
    
    if response.status_code != 200:
        print("Error from Huggingface API:", response.text)
        return "Error generating meme."

    result = response.json()
    full_text = result[0]['generated_text'] if isinstance(result, list) else result.get('generated_text', '')

    if not full_text:
        print("Empty response, retrying...")
        return make_it_funny(caption)

    if "[/INST]" in full_text:
        meme = full_text.split("[/INST]")[-1].strip()
    else:
        meme = full_text.strip()

    return meme

# --- Flask App ---
app = Flask(__name__)

@app.route('/generate-meme', methods=['POST'])
def generate_meme():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(BytesIO(file.read())).convert('RGB')
        caption = generate_caption(image)
        funny_meme = make_it_funny(caption)
        return jsonify({"caption": caption, "funny_meme": funny_meme})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Failed to process image"}), 500

# --- Main ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090, debug=True)