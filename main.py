import os
import time
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from tempfile import NamedTemporaryFile
import shutil

# ✅ Configure Gemini API
genai.configure(api_key="AIzaSyBwXy64k0asjYbPGSAms2oxitmTRxd2x0w")
model_name = "models/gemini-1.5-flash"

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS for frontend/backend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ PDF text extraction
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# ✅ Gemini prompt template
def create_gemini_prompt(pdf_text: str) -> str:
    return f"""
    Extract the following information from this report:

    - Title of the report
    - Date Visited
    - Visited By
    - Visited At
    - Key Inspections (important findings or activities)

    PDF Text:
    {pdf_text}
    """


def generate_gemini_response(prompt: str) -> str:
    model = genai.GenerativeModel(model_name=model_name)

    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            cleaned = response.text.replace("**", "").strip()  # remove markdown bold
            return cleaned
        except Exception as e:
            print(f"Gemini generation failed on attempt {attempt+1}: {e}")
            time.sleep(2)
    raise RuntimeError("Failed to generate Gemini response after 3 attempts.")


# ✅ Main API
@app.post("/summarize-pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    tmp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        text = extract_pdf_text(tmp_path)
        if not text:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF.")

        prompt = create_gemini_prompt(text)
        summary = generate_gemini_response(prompt)

        return JSONResponse(content={"summary": summary})
    except Exception as e:
        print("Exception in /summarize-pdf:", e)
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ✅ Health check
@app.get("/")
def home():
    return {"message": "PDF Summarizer using Gemini API is running."}
