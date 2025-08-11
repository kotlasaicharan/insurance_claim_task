import os
import json
import io
import logging
from datetime import datetime 

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from pdf2image import convert_from_bytes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enables CORS for all routes

try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    logger.info("✅ Gemini API configured successfully.")
except Exception as e:
    logger.error(f"❌ Failed to configure Gemini API: {e}")

SCHEMAS = {
    "medical_report": {
        "patient_name": "", "hospital_name": "",
        "report_date": "", "report_type": "", "clinical_findings": ""
    },
    
    "prescription": {
        "patient_name": "", "doctor_name": "", "clinic_name": "",
        "prescription_date": "",
        "diagnosis_notes": ""
    },
    "medical_bill": {
        "patient_name": "", "hospital_or_clinic_name": "",
        "bill_date": "",
        "bill_items": [], 
        "total_amount": ""
    }
}

def format_and_validate_date(date_string: str) -> str:
    """
    Attempts to parse a date string from various formats and returns it as 'dd/mm/yyyy'.
    Returns an empty string if the date is invalid or cannot be parsed.
    """
    if not isinstance(date_string, str) or not date_string.strip():
        return ""

    # A list of common date formats to try
    possible_formats = [
        '%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y',
        '%d %b %Y', '%d %B %Y', '%Y/%m/%d', '%d-%b-%y',
        '%m-%d-%Y', '%B %d, %Y',
    ]

    for fmt in possible_formats:
        try:
            date_obj = datetime.strptime(date_string, fmt)
            return date_obj.strftime('%d/%m/%Y')
        except ValueError:
            continue  # If parsing fails, try the next format

    logger.warning(f"Could not parse date: '{date_string}'. Returning empty string.")
    return "" # Return empty if no format matches


def convert_pdf_to_images(pdf_bytes):
    """Converts a PDF file in bytes to a list of PIL Image objects."""
    try:
        logger.info("Starting PDF conversion...")
        if not pdf_bytes:
            raise ValueError("Empty PDF bytes received")

        poppler_path = r"C:\poppler-24.08.0\Library\bin" # use your custom path
        if not os.path.exists(poppler_path):
            raise ValueError(f"Poppler not found at: {poppler_path}")
            
        logger.info(f"Using Poppler from: {poppler_path}")
        
        images = convert_from_bytes(
            pdf_bytes,
            poppler_path=poppler_path,
            dpi=200
        )
        
        if not images:
            raise ValueError("PDF conversion produced no images")
            
        logger.info(f"Successfully converted PDF to {len(images)} images")
        return images
    except Exception as e:
        logger.error(f"PDF conversion failed: {str(e)}", exc_info=True)
        return []
    

def extract_data_with_gemini(image_list, doc_type):

    if doc_type not in SCHEMAS:
        raise ValueError("Invalid document type specified.")
    
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    schema = SCHEMAS[doc_type]
    prompt = f"""
    Analyze the image of a '{doc_type}' and extract information into this exact JSON format.
    Ensure all date fields are formatted as dd/mm/yyyy.
    If a value is missing, not found, or ambiguous, return an empty string "" or an empty list [] as appropriate. Do not guess or fabricate information. Do not add explanations or markdown.
    Schema: {json.dumps(schema)}
    """
    
    extracted_pages = []
    for i, image in enumerate(image_list):
        logger.info(f"  > Processing page {i+1}/{len(image_list)} with Gemini...")
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            response = model.generate_content([prompt, {"mime_type": "image/png", "data": img_bytes}])
            response.resolve()
            
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
            page_data = json.loads(cleaned_text)

            # --- NEW: Post-processing to validate and format dates ---
            date_fields_to_check = ["report_date", "prescription_date", "bill_date"]
            for key, value in page_data.items():
                if key in date_fields_to_check:
                    page_data[key] = format_and_validate_date(value)
            
            extracted_pages.append(page_data)
        except Exception as e:
            logger.error(f"  > Gemini extraction failed for page {i+1}: {e}")
            extracted_pages.append(schema) # Append blank schema on failure

    return extracted_pages

@app.route('/')
def index():
    return send_file('index_2.html')

@app.route('/extract-documents', methods=['POST'])
def extract_documents():
    """
    [MODIFIED] Stage 1: Extracts page-wise data and includes filenames in the response.
    """
    logger.info("Received request for document extraction.")
    
    results = {"medical_report": [], "prescription": [], "medical_bill": []}
    doc_type_map = {
        "medical_report_files": "medical_report",
        "prescription_files": "prescription",
        "medical_bill_files": "medical_bill"
    }
    
    try:
        for field_name, doc_type in doc_type_map.items():
            uploaded_files = request.files.getlist(field_name)
            if not uploaded_files: continue
            
            logger.info(f"Processing {len(uploaded_files)} file(s) for type: {doc_type}")
            
            for file in uploaded_files:
                images = []
                if file.filename.lower().endswith('.pdf'):
                    images = convert_pdf_to_images(file.read())
                elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images = [Image.open(file.stream)]
                else:
                    logger.warning(f"Skipping unsupported file: {file.filename}")
                    continue
                
                if images:
                    # This now returns a list of page data
                    page_wise_data = extract_data_with_gemini(images, doc_type)
                    # We package it with the filename
                    results[doc_type].append({
                        "filename": file.filename,
                        "pages": page_wise_data
                    })

        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"An error occurred during extraction: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during document extraction."}), 500

@app.route('/adjudicate-claim', methods=['POST'])
def adjudicate_claim():
    logger.info("Received request for claim adjudication.")
    
    extracted_data = request.get_json()
    if not extracted_data:
        return jsonify({"error": "No extracted data provided."}), 400

    def summarize_category(category_data):
        if not category_data: return {}
        all_pages = category_data[0].get("pages", [])
        summary = {}
        for page in all_pages:
            for key, value in page.items():
                if value:
                    summary[key] = value
        return summary

    report_summary = summarize_category(extracted_data.get("medical_report", []))
    prescription_summary = summarize_category(extracted_data.get("prescription", []))
    bill_summary = summarize_category(extracted_data.get("medical_bill", []))
    
    expected_output = {
        "claimValidation": {
            "isPatientNameConsistent": True,
            "isConsistent": True,
            "isTreatmentBillMatch": True,
            "isDateSequenceLogical": True
        },
        "finalAssessment": {
            "isClaimValid": True,
            "confidenceScore":0.7,
            "reasoning": [
                "Example reasoning 1",
                "Example reasoning 2",
                "Example reasoning 3"
            ]
        }
    }


    adjudication_prompt = f"""
    You are an expert insurance claim adjudicator. Analyze the following structured data from a patient's claim and determine if it is valid. Some fields may be empty, indicating missing or inapplicable information, you can ignore those.

    **Clinical Report Data:**
    {json.dumps(report_summary, indent=2)}

    **Prescription Data:**
    {json.dumps(prescription_summary, indent=2)}

    **Billing Data:**
    {json.dumps(bill_summary, indent=2)}

Apply the following logic to generate your response.

isPatientNameConsistent: Only fail if there are substantial differences indicating different individuals (different first names, last names, or clearly unrelated identifiers).

isConsistent: Does the patient's diagnosis justify the prescribed medications?

isTreatmentBillMatch: Do all billed items (procedures, tests, medications) directly match what is documented in the clinical and prescription reports?

isDateSequenceLogical: Are all event dates in a logical chronological order (admission → treatment → discharge → billing)?

reasing: Every check must be justified with specific data from the source. This applies to both PASS and FAIL outcomes.
    Cite matching data points that confirm the check passed.
    Cite conflicting data points that prove the check failed.

    **Why this matters:** Explicit citations are mandatory to ensure conclusions are verifiable, prevent hallucinations, and support reliable human review.
    **Output Format:**
    {json.dumps(expected_output, indent=2)}
    
    Do not include any additional text or markdown formatting.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(adjudication_prompt)
        response.resolve()
        
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        adjudication_result = json.loads(cleaned_text)
        
        logger.info("Successfully adjudicated claim.")
        return jsonify(adjudication_result), 200
        
    except Exception as e:
        logger.error(f"An error occurred during adjudication: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during claim adjudication."}), 500

# --- MAIN EXECUTION ---
if __name__ == '__main__':

    app.run(debug=True, port=5001)
