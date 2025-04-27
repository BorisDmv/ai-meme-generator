from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont # <-- Import Pillow components
import torch
import os
import random
import io # <-- Import io for in-memory bytes handling
import base64 # <-- Import base64 for encoding image data
import textwrap

# --- Constants ---
FONT_PATH = "impact.ttf"
FONT_SIZE_RATIO = 0.08
TEXT_V_POSITION = 0.05 # Percentage from top for the *start* of the text block
TEXT_COLOR = "white"
STROKE_COLOR = "black"
STROKE_WIDTH_RATIO = 0.005
MAX_TEXT_WIDTH_RATIO = 0.90

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load BLIP model for image captioning
print("Loading BLIP image captioning model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load Mistral-7B model for humor generation
print("Loading Mistral-7B model for humor generation...")
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# --- Functions ---
def generate_caption(image):
    """Generate a caption for a given PIL image."""
    print("Thinking... Generating caption for the image...")
    # image = Image.open(image_path).convert('RGB') # No need to open again if passed as PIL obj
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    print("Caption generated:", caption)
    return caption


def random_instruction(caption):
    """Generate a slightly varied instruction to add randomness."""
    templates = [
        f"Write a funny one-line meme about: {caption}. Keep it very short, under 10 words.",
        f"Make a hilarious meme caption for: {caption}",
        f"Create a very short (under 10 words) funny meme caption for: {caption}",
        f"Come up with a meme for: {caption}",
        f"Write a clever meme line for: {caption}",
        f"Invent a funny short meme about: {caption}",
    ]
    return random.choice(templates)


def make_it_funny(caption, max_retries=2):
    """Generate a funny meme caption using Mistral-7B with retries and better cleaning."""
    print(f"Attempting to generate meme for caption: '{caption}'")

    for attempt in range(max_retries):
        print(f"Generation attempt {attempt + 1}/{max_retries}")
        instruction = random_instruction(caption)
        prompt = f"<s>[INST] {instruction} [/INST]"
        print(f"Using prompt: {prompt}")

        inputs = mistral_tokenizer(prompt, return_tensors="pt", padding=True)

        with torch.no_grad():
            output = mistral_model.generate(
                **inputs,
                # Slightly increase max_new_tokens to give more room initially
                max_new_tokens=50,
                do_sample=True,
                temperature=0.75, # Slightly adjusted temperature
                top_p=0.9,
                top_k=50,
                pad_token_id=mistral_tokenizer.eos_token_id,
                # Add repetition penalty to discourage boring output
                repetition_penalty=1.1
            )

        # Decode the full output first for debugging
        full_decoded_output = mistral_tokenizer.decode(output[0], skip_special_tokens=False)
        print(f"Attempt {attempt + 1} - Full raw output (incl. special tokens): {full_decoded_output}")

        # Decode skipping special tokens for processing
        decoded_output = mistral_tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Attempt {attempt + 1} - Decoded output (no special tokens): {decoded_output}")

        # --- Improved Cleaning and Extraction ---
        funny_meme = ""
        # 1. Try splitting by [/INST]
        parts = decoded_output.split('[/INST]')
        if len(parts) > 1:
            funny_meme_raw = parts[-1].strip()
            print(f"Attempt {attempt + 1} - Text after [/INST]: '{funny_meme_raw}'")
        else:
             # 2. Fallback: If [/INST] isn't there, maybe it just generated text after the prompt?
             # Remove the original prompt text from the decoded output
             # Use re.escape to handle special characters in the instruction if any
             instruction_pattern = re.escape(instruction)
             # Try to find text *after* the instruction part of the prompt
             match = re.search(f'{instruction_pattern}\\s*\[/INST\]\\s*(.*)', decoded_output, re.DOTALL | re.IGNORECASE)
             if match:
                 funny_meme_raw = match.group(1).strip()
                 print(f"Attempt {attempt + 1} - Text found via regex after instruction: '{funny_meme_raw}'")
             else:
                # Last resort: If prompt wasn't found either, maybe just take the end?
                # This is less reliable.
                funny_meme_raw = decoded_output # Keep the whole thing for now, clean later
                print(f"Attempt {attempt + 1} - Couldn't find [/INST] or instruction pattern reliably.")


        # 3. Clean common model prefixes/artefacts (add more as you see them)
        prefixes_to_remove = [
            "Funny meme:", "Meme text:", "Here's a meme:", "Sure, here you go:",
            "Okay, here's a short meme:", "\"", "'"
        ]
        for prefix in prefixes_to_remove:
             # Case-insensitive removal
            if funny_meme_raw.lower().startswith(prefix.lower()):
                funny_meme_raw = funny_meme_raw[len(prefix):].strip()

        # Remove potential trailing quotes or incomplete sentences
        funny_meme_raw = funny_meme_raw.strip(' "\'.')

        # 4. Apply word limit and uppercase
        if funny_meme_raw:
            funny_meme_words = funny_meme_raw.split()
            # Ensure it has at least 2 words to be considered valid
            if len(funny_meme_words) >= 2:
                 funny_meme = ' '.join(funny_meme_words[:10]).upper() # Keep the word limit
                 print(f"Attempt {attempt + 1} - Cleaned and formatted meme: '{funny_meme}'")
                 # If we got a valid meme, break the retry loop
                 break
            else:
                print(f"Attempt {attempt + 1} - Generated text too short after cleaning: '{funny_meme_raw}'")
        else:
            print(f"Attempt {attempt + 1} - No text extracted after cleaning.")

    # --- Final Check after Retries ---
    if not funny_meme:
        print("Failed to generate a valid meme after all attempts. Returning fallback.")
        funny_meme = "I GOT NUTHIN'..." # Default fallback message

    print(f"Final meme text being returned: '{funny_meme}'")
    return funny_meme


def draw_text_on_image(image, text):
    """Draws the meme text onto the PIL image, with text wrapping."""
    print("Drawing text on image with wrapping...")
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size

    # --- Font Loading ---
    font_size = int(image_height * FONT_SIZE_RATIO)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        print(f"Warning: Font file not found at {FONT_PATH}. Using default font.")
        # Pillow's default font is very small and doesn't support truetype features well
        # Consider ensuring the font file IS present or providing a guaranteed fallback path
        try:
            # Try loading a known system font as a slightly better fallback
             font = ImageFont.truetype("arial.ttf", font_size) # Example: Arial on Windows/some Linux
        except IOError:
             font = ImageFont.load_default() # Absolute fallback

    # --- Text Wrapping ---
    max_text_pixel_width = int(image_width * MAX_TEXT_WIDTH_RATIO)
    stroke_width = max(1, int(font_size * STROKE_WIDTH_RATIO))

    # Check if text needs wrapping at all using textbbox for better accuracy
    try:
        # Get bounding box of the original single-line text
        original_bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        original_text_width = original_bbox[2] - original_bbox[0]
    except AttributeError:
         # Fallback for older Pillow versions that might not have stroke_width in textbbox
         print("Warning: Pillow version might be old. Using textlength for width check.")
         original_text_width = draw.textlength(text, font=font)


    if original_text_width <= max_text_pixel_width:
        # Text fits on one line
        print("Text fits on a single line.")
        wrapped_text = text
        # Use multiline_textbbox even for single line for consistency in positioning calculation
        try:
            final_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, stroke_width=stroke_width, align="center")
        except AttributeError:
             final_bbox = draw.textbbox((0, 0), wrapped_text, font=font) # Fallback

    else:
        # Text needs wrapping
        print("Text requires wrapping.")
        # Estimate average character width to guide textwrap (approximation)
        avg_char_width = original_text_width / len(text) if len(text) > 0 else font_size * 0.5 # Estimate if text empty
        # Calculate wrap width in characters
        wrap_width_chars = int(max_text_pixel_width / avg_char_width) if avg_char_width > 0 else 10
        wrap_width_chars = max(1, wrap_width_chars) # Ensure at least 1 character wide

        print(f"Calculated wrap width: {wrap_width_chars} characters (approx)")

        # Wrap the text
        wrapper = textwrap.TextWrapper(width=wrap_width_chars, break_long_words=True)
        lines = wrapper.wrap(text)
        wrapped_text = '\n'.join(lines)
        print(f"Wrapped text:\n{wrapped_text}")

        # Calculate bounding box for the wrapped text
        try:
             final_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, stroke_width=stroke_width, align="center")
        except AttributeError:
             # Fallback for older Pillow
             final_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")


    # --- Positioning and Drawing ---
    final_text_width = final_bbox[2] - final_bbox[0]
    final_text_height = final_bbox[3] - final_bbox[1]

    # Calculate top-left position (x, y) for the text block to be centered horizontally
    x = (image_width - final_text_width) / 2
    y = image_height * TEXT_V_POSITION # Start drawing from this vertical position

    print(f"Drawing text block at (x={x:.2f}, y={y:.2f}) with width={final_text_width}, height={final_text_height}")

    # Use multiline_text for drawing (works for single and multiple lines)
    draw.multiline_text(
        (x, y),
        wrapped_text,
        font=font,
        fill=TEXT_COLOR,
        stroke_width=stroke_width,
        stroke_fill=STROKE_COLOR,
        align="center" # Center align lines within the text block
    )

    print("Text drawing complete.")
    return image

def image_to_base64(image, format="PNG"):
    """Converts a PIL image to a Base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}" # Add data URI prefix


# --- Flask Routes ---
@app.route('/generate-meme', methods=['POST'])
def generate_meme_route(): # Renamed function to avoid conflict
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # --- Load image using Pillow ---
        image = Image.open(file.stream).convert('RGB') # Read directly from stream

        # --- Generate caption and funny meme ---
        caption = generate_caption(image.copy()) # Pass PIL image, use copy if needed
        funny_meme_text = make_it_funny(caption)

        # --- Draw text onto the image ---
        meme_image = draw_text_on_image(image, funny_meme_text) # image is modified in-place

        # --- Convert final image to Base64 ---
        base64_image = image_to_base64(meme_image, format="PNG") # Save as PNG for better text quality

        # --- Return the result ---
        return jsonify({
            "caption": caption,
            "funny_meme_text": funny_meme_text, # Still send text if needed
            "meme_image_base64": base64_image # Send the image data
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        # Provide more specific error feedback if possible
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500
    # Removed temporary file saving/deleting as we process in memory


# --- Main ---
if __name__ == "__main__":
    # Create temp dir if it doesn't exist (though not used anymore)
    # if not os.path.exists('temp'):
    #     os.makedirs('temp')
    app.run(host="0.0.0.0", port=9090, debug=True)