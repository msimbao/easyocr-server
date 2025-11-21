from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import easyocr
import io
from PIL import Image
import fitz  # PyMuPDF
from docx import Document
import numpy as np
from typing import List
import uvicorn

app = FastAPI(title="EasyOCR Service", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR reader (English by default, add more languages as needed)
reader = None

@app.on_event("startup")
async def startup_event():
    global reader
    # Initialize with English, add more languages like ['en', 'es', 'fr'] if needed
    reader = easyocr.Reader(['en'], gpu=False)

@app.get("/")
async def root():
    return {"message": "EasyOCR Service is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "reader_loaded": reader is not None}

def extract_text_from_image(image_bytes: bytes) -> List[str]:
    """Extract text from image using EasyOCR"""
    img = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(img)
    results = reader.readtext(img_array)
    # Extract just the text from results (results contain bbox, text, confidence)
    return [detection[1] for detection in results]

def extract_text_from_pdf(pdf_bytes: bytes) -> List[str]:
    """Extract text from PDF - tries text extraction first, then OCR on images"""
    all_text = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        
        # Try direct text extraction first
        text = page.get_text()
        if text.strip():
            all_text.append(text)
        else:
            # If no text, render page as image and OCR it
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img_bytes = pix.tobytes("png")
            ocr_text = extract_text_from_image(img_bytes)
            all_text.extend(ocr_text)
    
    return all_text

def extract_text_from_docx(docx_bytes: bytes) -> List[str]:
    """Extract text from Word document"""
    doc = Document(io.BytesIO(docx_bytes))
    return [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]

def extract_text_from_txt(txt_bytes: bytes) -> List[str]:
    """Extract text from plain text file"""
    try:
        text = txt_bytes.decode('utf-8')
        return [line for line in text.split('\n') if line.strip()]
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        text = txt_bytes.decode('latin-1')
        return [line for line in text.split('\n') if line.strip()]

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from uploaded file (image, PDF, DOCX, or TXT)
    
    Supported formats:
    - Images: jpg, jpeg, png, bmp, tiff
    - PDF: pdf
    - Word: docx
    - Text: txt
    """
    try:
        contents = await file.read()
        filename_lower = file.filename.lower()
        
        # Determine file type and process accordingly
        if filename_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
            extracted_text = extract_text_from_image(contents)
            file_type = "image"
            
        elif filename_lower.endswith('.pdf'):
            extracted_text = extract_text_from_pdf(contents)
            file_type = "pdf"
            
        elif filename_lower.endswith('.docx'):
            extracted_text = extract_text_from_docx(contents)
            file_type = "docx"
            
        elif filename_lower.endswith('.txt'):
            extracted_text = extract_text_from_txt(contents)
            file_type = "txt"
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: images (jpg, png, etc.), pdf, docx, txt"
            )
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "file_type": file_type,
            "text": extracted_text,
            "text_joined": "\n".join(extracted_text),
            "lines_count": len(extracted_text)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)