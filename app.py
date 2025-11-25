from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import cv2

app = Flask(__name__)

# Load models
det_session = ort.InferenceSession("ch_ppocr_mobile_v2.0_det.onnx")
rec_session = ort.InferenceSession("ch_ppocr_mobile_v2.0_rec.onnx")

# Character dictionary for Chinese + English
CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"  # Simplified - expand with full char set

def preprocess_detection(img):
    """Preprocess image for text detection"""
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    # Resize to model input size (e.g., 640x640)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # Normalize
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, 0)  # Add batch dimension
    
    return img, h, w

def preprocess_recognition(img):
    """Preprocess cropped text region for recognition"""
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Resize to recognition model input (e.g., 32x320)
    img = cv2.resize(img, (320, 32))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    
    return img

def detect_text(img):
    """Run text detection"""
    processed_img, orig_h, orig_w = preprocess_detection(img)
    
    input_name = det_session.get_inputs()[0].name
    output = det_session.run(None, {input_name: processed_img})[0]
    
    # Post-process detection output to get bounding boxes
    # This is simplified - actual implementation needs proper box extraction
    boxes = []
    
    # Threshold and find contours from detection map
    det_map = output[0, 0]
    det_map = (det_map > 0.3).astype(np.uint8) * 255
    det_map = cv2.resize(det_map, (orig_w, orig_h))
    
    contours, _ = cv2.findContours(det_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Filter small boxes
            boxes.append([x, y, x+w, y+h])
    
    return boxes

def recognize_text(img, box):
    """Recognize text in a bounding box"""
    x1, y1, x2, y2 = box
    cropped = img.crop((x1, y1, x2, y2))
    
    processed_img = preprocess_recognition(cropped)
    
    input_name = rec_session.get_inputs()[0].name
    output = rec_session.run(None, {input_name: processed_img})[0]
    
    # Decode CTC output
    pred_indices = np.argmax(output, axis=2)[0]
    
    # Remove duplicates and blanks (simple CTC decoding)
    text = ""
    prev_idx = -1
    for idx in pred_indices:
        if idx != prev_idx and idx > 0 and idx < len(CHARS):
            text += CHARS[idx]
        prev_idx = idx
    
    return text

@app.route('/ocr', methods=['POST'])
def ocr():
    """OCR endpoint - accepts image file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    try:
        # Load image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Detect text regions
        boxes = detect_text(img)
        
        # Recognize text in each box
        results = []
        for box in boxes:
            text = recognize_text(img, box)
            results.append({
                'box': box,
                'text': text
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)