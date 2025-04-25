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
    print("Thinking... Generating funny meme caption...")
    prompt = f"Turn this into a funny meme caption: {caption}\nMeme:"
    # Tokenize the prompt with padding enabled
    inputs = gpt_tokenizer(prompt, return_tensors="pt", padding=True)

    with torch.no_grad():
        # Use inputs directly in the generate function (no need to pass attention_mask)
        output = gpt_model.generate(
            **inputs,  # pass all inputs as is
            max_length=200,  # Increased length for more content
            do_sample=True,
            temperature=0.7,  # Lower temperature to make it more predictable
            pad_token_id=gpt_tokenizer.eos_token_id  # Explicitly set pad_token_id
        )

    meme = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
    print("Full meme generated:", meme)  # Log the full output
    # Extract the part after "Meme:"
    funny_meme = meme.split("Meme:")[-1].strip()
    print("Funny meme text generated:", funny_meme)
    return funny_meme


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_meme_text.py path_to_image")
    else:
        print("Starting meme generation process...")
        caption = generate_caption(sys.argv[1])
        funny = make_it_funny(caption)
        print("Final Funny Meme Text:", funny)
