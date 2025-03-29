from flask import Flask, request, jsonify
import numpy as np
import cv2
import pytesseract
import json
import os
from tensorflow.keras.models import load_model
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)


model = load_model('doc_classifier.h5')
CLASSES_LIST = ["Aadhar", "Passport", "Licence", "Pancard"]


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

pytesseract.pytesseract.tesseract_cmd = r"./Tesseract-OCR/tesseract.exe"  



# Image Preprocessing Function 
def sharpen_image(image):
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, sharpen_kernel)



# OCR Function using Tesseract
def img_to_txt(img):
    return pytesseract.image_to_string(img, lang="eng+mar+hin")



def structured(doc_type, text):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You extract structured data from documents and return raw JSON only."
            },
            {
                "role": "user",
                "content": f"Extract structured information from this {doc_type} document:\n{text}\n"
                           f"Return only a valid JSON object without any formatting, explanations, or extra characters. "
                           f"Ensure the JSON contains exactly:\n"
                           f'{{"Name": "Person Name", "Date of Birth": "DD/MM/YYYY", "ID Number": "ID Value", "Document Name": "{doc_type}"}}'
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

# Combined API Endpoint (Classification + Extraction)
@app.route('/process', methods=['POST'])
def process_document():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']

        # Read image
        image = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        #  Document Classification
        img_resized = cv2.resize(img, (224, 224)) / 255.0  
        img_resized = np.expand_dims(img_resized, axis=0)  
        prediction = model.predict(img_resized)
        predicted_class = CLASSES_LIST[np.argmax(prediction)]
        confidence = float(np.max(predictTesseract-OCRion))

        # Text Extraction & Structuring
        img_sharpened = sharpen_image(img)  
        extracted_text = img_to_txt(img_sharpened)
        structured_data = structured(predicted_class, extracted_text)
        extracted_json = json.loads(structured_data)


        return jsonify({
            "predicted_class": predicted_class,
            "extracted_data": extracted_json
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
