from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from google import genai
import base64
import mimetypes
import os

app = FastAPI()

# Load API key from environment variable
API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize the Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    api_key=API_KEY
)

def encode_image(file: UploadFile):
    return base64.b64encode(file.file.read()).decode("utf-8")


@app.post("/generate-vto")
async def generate_vto(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...)
):
    try:
        # Encode images to base64
        person_b64 = encode_image(person_image)
        clothing_b64 = encode_image(clothing_image)

        person_type = mimetypes.guess_type(person_image.filename)[0]
        clothing_type = mimetypes.guess_type(clothing_image.filename)[0]

        # Send content to Gemini
        response = model.generate_content([
            {
                "mime_type": person_type,
                "data": base64.b64decode(person_b64),
            },
            {
                "mime_type": clothing_type,
                "data": base64.b64decode(clothing_b64),
            },
            {
                "text": "Generate a realistic full-body photo showing the person wearing the clothing."
            }
        ])

        # Get image from response
        image_parts = [
            part for part in response.parts
            if hasattr(part, "inline_data") and part.inline_data.data
        ]

        if not image_parts:
            return JSONResponse(content={"error": "No image returned by Gemini."}, status_code=500)

        image_data = image_parts[0].inline_data.data
        mime_type = image_parts[0].inline_data.mime_type

        base64_str = base64.b64encode(image_data).decode("utf-8")

        return {
            "mime_type": mime_type,
            "base64_image": base64_str,
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
