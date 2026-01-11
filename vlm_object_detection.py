from groq import Groq
import base64
import os
from pathlib import Path

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Initialize Groq client
# Get API key from environment variable or use the one provided
api_key = os.environ.get("GROQ_API_KEY") or ""
client = Groq(api_key=api_key)

# Test folder path
test_folder = Path("test_frame")

# Get all image files from test_frame
image_files = list(test_folder.glob("*.png")) + list(test_folder.glob("*.jpg")) + list(test_folder.glob("*.jpeg"))

if not image_files:
    print(f"No images found in {test_folder}")
    exit(1)

print(f"Found {len(image_files)} image(s) to test\n")
print("=" * 70)

# Encode all images
print("Encoding images...")
encoded_images = []
for image_path in image_files:
    try:
        base64_image = encode_image(image_path)
        encoded_images.append((image_path.name, base64_image))
        print(f"  ✓ Encoded: {image_path.name}")
    except Exception as e:
        print(f"  ✗ Error encoding {image_path.name}: {str(e)}")

if not encoded_images:
    print("No images could be encoded. Exiting.")
    exit(1)

print(f"\nSending {len(encoded_images)} image(s) to VLM in a single request...")
print("-" * 70)

try:
    # Build content array with text prompt and all images
    content = [
        {
            "type": "text", 
            "text": f"These {len(encoded_images)} images are sequential video frames extracted from a live RTSP stream. A YOLO object detection model has flagged these frames as potential weapon detections. Your task is to verify if weapons (guns, knives, firearms, or any weapons) are actually present in these frames as evidence from the live stream.\n\nAnalyze each frame carefully in the context of being part of a continuous live video stream. For each frame, determine:\n- 'YES' if a weapon is clearly visible and confirmed\n- 'NO' if no weapon is found (false positive from YOLO)\n\nProvide the frame filename and a brief explanation for each frame. Consider the temporal context - these frames are evidence from the same live stream, so analyze them as a sequence of evidence, not as separate unrelated images."
        }
    ]
    
    # Add all images to the content
    for image_name, base64_image in encoded_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
            },
        })
    
    # Send single request with all images
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    
    response = chat_completion.choices[0].message.content
    print(f"\nVLM Response:\n{response}\n")
    
except Exception as e:
    print(f"Error processing images: {str(e)}\n")

print("=" * 70)
print("Testing complete!")
