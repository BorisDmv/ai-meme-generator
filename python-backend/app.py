from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

# Load BLIP model for captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load GPT-Neo model for humor
gpt_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token  # Set pad_token to eos_token
gpt_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

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
    prompt = f"Create a short, funny meme caption for this: {caption}. Make it snappy and clever. Meme caption:"
    
    inputs = gpt_tokenizer(prompt, return_tensors="pt", padding=True)

    if gpt_tokenizer.pad_token is None:
        gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None

    with torch.no_grad():
        output = gpt_model.generate(
            inputs['input_ids'],
            max_new_tokens=30,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            top_k=40,
            num_return_sequences=1,
            pad_token_id=gpt_tokenizer.pad_token_id,
            attention_mask=attention_mask
        )

    meme = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
    meme = meme.split("Meme caption:")[-1].strip()

    for delimiter in [".", "!", "?"]:
        if delimiter in meme:
            meme = meme.split(delimiter)[0] + delimiter
            break

    # 🛠️ Optional: Also remove weird starting phrases (if needed)
    if meme.startswith("If I'm looking at a picture of"):
        meme = meme.split("If I'm looking at a picture of")[1].strip()

    return meme.strip()



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_meme_text.py path_to_image")
    else:
        print("Starting meme generation process...")
        caption = generate_caption(sys.argv[1])
        funny = make_it_funny(caption)
        print("Final Funny Meme Text:", funny)
