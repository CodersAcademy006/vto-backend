from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from google import genai
import base64
import mimetypes
import os

app = FastAPI()

# Set your Gemini API key via environment variable (Render will handle this)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def encode_image(file: UploadFile):
    return base64.b64encode(file.file.read()).decode("utf-8")


@app.post("/generate-vto")
async def generate_vto(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...)
):
    try:
        # Encode images
        person_b64 = encode_image(person_image)
        clothing_b64 = encode_image(clothing_image)

        person_type = mimetypes.guess_type(person_image.filename)[0]
        clothing_type = mimetypes.guess_type(clothing_image.filename)[0]

        model = genai.GenerativeModel("gemini-1.5-pro")

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
                "text": "Visualize the person wearing this clothing as a full-body output photo, realistic VTO."
            }
        ])

        # Handle response
        image_parts = [
            part for part in response.parts
            if hasattr(part, "inline_data") and part.inline_data.data
        ]

        if not image_parts:
            return JSONResponse(content={"error": "No image in response"}, status_code=500)

        image_data = image_parts[0].inline_data.data
        mime_type = image_parts[0].inline_data.mime_type

        base64_str = base64.b64encode(image_data).decode("utf-8")
        return {
            "mime_type": mime_type,
            "base64_image": base64_str,
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)