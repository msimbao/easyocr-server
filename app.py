from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
import PyPDF2
import docx
import io
import os
from PIL import Image
import pdf2image
import numpy as np

app = Flask(__name__)
CORS(app)

# Initialize EasyOCR reader (loads once at startup)
print("Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if available
print("EasyOCR model loaded successfully!")

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'EasyOCR Server Running',
        'endpoints': {
            'extract': 'POST /api/extract - Upload file for text extraction'
        }
    })

# Main extraction endpoint
@app.route('/api/extract', methods=['POST'])
def extract_text():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = file.filename
        file_bytes = file.read()
        mimetype = file.content_type
        
        print(f"Processing file: {filename} ({mimetype})")
        
        extracted_text = ''
        
        # Handle different file types
        if mimetype.startswith('image/'):
            # Process images with EasyOCR
            image = Image.open(io.BytesIO(file_bytes))
            image_np = np.array(image)
            
            result = reader.readtext(image_np)
            # Extract text from result tuples (bbox, text, confidence)
            extracted_text = '\n'.join([text for (bbox, text, conf) in result])
        
        elif mimetype == 'application/pdf':
            # First try to extract text from PDF
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text_from_pdf = ''
            
            for page in pdf_reader.pages:
                text_from_pdf += page.extract_text()
            
            # If PDF has substantial text, use it
            if len(text_from_pdf.strip()) > 50:
                extracted_text = text_from_pdf
            else:
                # PDF is likely scanned, use OCR
                print("PDF appears to be scanned, using OCR...")
                images = pdf2image.convert_from_bytes(file_bytes)
                
                for i, image in enumerate(images):
                    print(f"Processing page {i+1}/{len(images)}")
                    image_np = np.array(image)
                    result = reader.readtext(image_np)
                    page_text = '\n'.join([text for (bbox, text, conf) in result])
                    extracted_text += f"\n--- Page {i+1} ---\n{page_text}\n"
        
        elif mimetype in [
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword'
        ]:
            # Extract text from Word documents
            doc = docx.Document(io.BytesIO(file_bytes))
            extracted_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        elif mimetype == 'text/plain':
            # Plain text files
            extracted_text = file_bytes.decode('utf-8')
        
        else:
            return jsonify({
                'error': 'Unsupported file type',
                'supportedTypes': ['images (jpg, png, etc.)', 'PDF', 'Word (docx)', 'Text files']
            }), 400
        
        return jsonify({
            'success': True,
            'text': extracted_text,
            'filename': filename,
            'mimetype': mimetype,
            'textLength': len(extracted_text)
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': 'Failed to extract text',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
