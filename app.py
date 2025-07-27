import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import base64
import io
# Import load_dotenv ONLY if you use a .env file for local development
# from dotenv import load_dotenv
from google.cloud import storage # Import Google Cloud Storage client at the top level

# If you use a .env file locally, uncomment the next line:
# load_dotenv()

# --- Flask App Setup ---
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# --- Configuration ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
# Model will be downloaded to the root directory during Render build
MODEL_FILENAME = 'crop_diagnosis_best_model.keras'
CLASS_INDICES_FILENAME = os.path.join(os.getcwd(), 'class_indices.json')
# Path for local Firebase service account key (NOT for Render deployment)
LOCAL_FIREBASE_KEY_PATH = os.path.join(os.getcwd(), 'serviceAccountKey.json')

# --- GCS Configuration for Model Download ---
# REPLACE with YOUR ACTUAL GCS BUCKET NAME
GCS_BUCKET_NAME = "crop-doctor-ml-models-aayush-2025"
# Name of the model file as it is stored in your GCS bucket
GCS_MODEL_BLOB_NAME = "crop_diagnosis_best_model.keras"

# --- API Key (Read from environment variable for security) ---
# This will read from Render's environment variables or your local .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Global variables for model and class labels ---
model = None
class_labels = None
db = None # Firestore database client

# --- Function to download model from GCS ---
def download_model_from_gcs():
    # Check if model already exists (e.g., if running locally after first download)
    if os.path.exists(MODEL_FILENAME):
        print(f"Model already exists at {MODEL_FILENAME}. Skipping GCS download.")
        return

    try:
        # Authenticate using environment variable (for Render deployment)
        # This env var (GCS_SERVICE_ACCOUNT_KEY) will contain the JSON string of your service account key
        gcs_key_json = os.getenv("GCS_SERVICE_ACCOUNT_KEY")
        if gcs_key_json:
            credentials_info = json.loads(gcs_key_json)
            client = storage.Client.from_service_account_info(credentials_info)
            print("Using GCS_SERVICE_ACCOUNT_KEY environment variable for GCS authentication.")
        else:
            # Fallback for local testing if you have default GCS credentials configured
            # (e.g., via `gcloud auth application-default login`) or if bucket is public.
            print("Warning: GCS_SERVICE_ACCOUNT_KEY not set. Attempting default GCS authentication for model download.")
            client = storage.Client()

        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_MODEL_BLOB_NAME)

        print(f"Downloading model from gs://{GCS_BUCKET_NAME}/{GCS_MODEL_BLOB_NAME} to {MODEL_FILENAME}...")
        blob.download_to_filename(MODEL_FILENAME)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model from GCS: {e}")
        # Re-raise the exception to cause the Render build to fail if download fails
        raise

# --- Initialize Firebase Admin SDK ---
def initialize_firebase():
    global db
    print("Attempting to initialize Firebase...") # Added for debugging
    try:
        # For Render deployment, Firebase credentials should be passed as an environment variable (FIREBASE_CONFIG_JSON)
        firebase_config_json = os.getenv("FIREBASE_CONFIG_JSON")
        if firebase_config_json:
            cred = credentials.Certificate(json.loads(firebase_config_json))
            print("Using FIREBASE_CONFIG_JSON environment variable for Firebase initialization.")
        elif os.path.exists(LOCAL_FIREBASE_KEY_PATH): # Fallback for local development
            print("Using local serviceAccountKey.json for Firebase initialization.")
            cred = credentials.Certificate(LOCAL_FIREBASE_KEY_PATH)
        else:
            print(f"Error: Firebase service account key not found at {LOCAL_FIREBASE_KEY_PATH} or FIREBASE_CONFIG_JSON not set.")
            print("Please ensure serviceAccountKey.json is in project root (local) or FIREBASE_CONFIG_JSON env var is set (Render).")
            return False

        if not firebase_admin._apps: # Check if app is already initialized to prevent errors
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print(f"Firebase initialized successfully. db object is: {db}") # Added for debugging
        return True
    except Exception as e:
        print(f"Failed to initialize Firebase: {e}")
        import traceback # Added for debugging
        traceback.print_exc() # Added for debugging
        return False

# --- Load Model and Class Indices on startup ---
def load_resources():
    global model, class_labels
    print("Attempting to load model and class indices...") # Added for debugging
    try:
        # The model should have been downloaded by the Render build command before this runs.
        if not os.path.exists(MODEL_FILENAME):
            print(f"Error: Model file not found at {MODEL_FILENAME} during load_resources. It should have been downloaded by build command.")
            return False
        if not os.path.exists(CLASS_INDICES_FILENAME):
            print(f"Error: Class indices file not found at {CLASS_INDICES_FILENAME}. Please ensure it's in the project root.")
            return False

        model = load_model(MODEL_FILENAME)
        with open(CLASS_INDICES_FILENAME, 'r') as f:
            class_indices = json.load(f)
        class_labels = {v: k for k, v in class_indices.items()}
        print("Model and class indices loaded successfully.")
        return True
    except Exception as e:
        print(f"Failed to load model or class indices: {e}")
        import traceback # Added for debugging
        traceback.print_exc() # Added for debugging
        return False

# --- Call initialization functions directly when the module is imported ---
# These lines will run when Gunicorn imports app.py
if not initialize_firebase():
    print("CRITICAL ERROR: Firebase initialization failed during app startup.")
    # You might want to raise an exception or exit here in a real production app
    # to prevent the app from running in a broken state.
    # For now, we'll let it proceed to allow other debugging.

if not load_resources():
    print("CRITICAL ERROR: Model and class indices loading failed during app startup.")
    # Similar to Firebase, consider raising an exception or exiting.


# --- Gemini Integration ---
def get_gemini_diagnosis(disease_name, user_context):
    if not GEMINI_API_KEY:
        return "Error: Gemini API key not configured on the backend. Please set the GEMINI_API_KEY environment variable."

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
        Act as an expert agronomist and plant pathologist for a user in India.

        *Primary Diagnosis from Image Analysis:*
        The image analysis model has identified the plant disease as: "{disease_name.replace('_', ' ')}".

        *Additional Context from the User (Detailed Questionnaire):*
        - Plant Symptoms:
            - Leaf discoloration observed: "{user_context.get('leaf_discoloration', 'Not specified.')}"
            - Wilting or dropping: "{user_context.get('wilting_dropping', 'Not specified.')}"
        - Environmental Conditions:
            - Recent weather: "{user_context.get('recent_weather', 'Not specified.')}"
            - Temperature condition: "{user_context.get('temperature_condition', 'Not specified.')}"
        - Treatment History:
            - Recent fertilizer application: "{user_context.get('recent_fertilizer', 'Not specified.')}"
            - Previous pesticide use: "{user_context.get('previous_pesticide', 'Not specified.')}"
        - Pest Observations:
            - Insects observed: "{user_context.get('insects_observed', 'Not specified.')}"
            - Evidence of pest damage: "{user_context.get('evidence_of_damage', 'Not specified.')}"
        - Plant Management:
            - Watering frequency: "{user_context.get('watering_frequency', 'Not specified.')}"
            - Plant age/growth stage: "{user_context.get('plant_age_growth', 'Not specified.')}"

        *Your Task:*
        Based on all the information above, provide a comprehensive and actionable report. Structure your response with the following sections using clear markdown:

        1. *Integrated Diagnosis*
        2. *Immediate Action Plan (Organic)*
        3. *Immediate Action Plan (Chemical)*
        4. *Long-Term Prevention Strategy*
        5. *Local Agricultural Support (India)*
        """

        response = gemini_model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"An error occurred with the Gemini API: {e}"

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not class_labels:
        return jsonify({"error": "Model not loaded. Please ensure model and class indices files exist."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img_bytes = file.read()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        img_stream = io.BytesIO(img_bytes)
        
        img = image.load_img(img_stream, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100
        predicted_class_name = class_labels[predicted_class_index]

        if db:
            history_ref = db.collection('predictions')
            history_ref.add({
                'timestamp': firestore.SERVER_TIMESTAMP,
                'image_base64': img_base64,
                'predicted_class_name': predicted_class_name,
                'confidence': float(confidence)
            })

        return jsonify({
            "predicted_class_name": predicted_class_name,
            "confidence": float(confidence)
        })
    except Exception as e:
        print(f"ERROR during prediction: {e}") # Added for debugging
        import traceback # Added for debugging
        traceback.print_exc() # Added for debugging
        return jsonify({"error": f"Error during prediction: {e}"}), 500

# --- Get Diagnosis Endpoint ---
@app.route('/get_diagnosis', methods=['POST'])
def get_diagnosis():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request data"}), 400

    disease_name = data.get('disease_name')
    user_context = data.get('user_context', {})

    if not disease_name:
        return jsonify({"error": "Disease name is required for diagnosis"}), 400

    final_report = get_gemini_diagnosis(disease_name, user_context)
    return jsonify({"report": final_report})

# --- History Endpoint ---
@app.route('/history', methods=['GET'])
def get_history():
    if not db:
        print("ERROR: Firestore 'db' object is None before history fetch.") # Added for debugging
        return jsonify({"error": "Firestore not initialized."}), 500

    try:
        predictions_ref = db.collection('predictions').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50)
        docs = predictions_ref.stream()

        history_data = []
        for doc in docs:
            data = doc.to_dict()
            if 'timestamp' in data and data['timestamp']:
                data['timestamp'] = data['timestamp'].isoformat()
            history_data.append(data)

        return jsonify({"history": history_data})
    except Exception as e:
        print(f"ERROR fetching history: {e}") # Added for debugging
        import traceback # Added for debugging
        traceback.print_exc() # Added for debugging
        return jsonify({"error": f"Error fetching history: {e}"}), 500

# --- Frontend Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/history_page')
def history_page():
    return render_template('history.html')

@app.route('/user_guide')
def user_guide():
    return render_template('user.html') # Corrected from user_guide.html

@app.route('/tools')
def tools_page():
    return render_template('tools.html')

# --- Main entry point for Flask (only for local development) ---
if __name__ == '__main__':
    # For local development, you might call download_model_from_gcs() here
    # if you want it to download every time you run app.py locally.
    # For Render, it will be handled by the build command.
    
    # It's crucial to call download_model_from_gcs() before load_resources()
    # when running locally, if the model isn't already present.
    # For Render, the build command handles the download.
    
    # Example for local run (uncomment if you want app.py to manage local download):
    # try:
    #     download_model_from_gcs()
    # except Exception as e:
    #     print(f"Local model download failed: {e}")
    #     # Decide if you want to exit or continue without model
    #     exit(1) # Exit if model download is critical

    # These are now called outside this block for Gunicorn compatibility
    # if initialize_firebase() and load_resources():
    print("Starting Flask server for local development...")
    app.run(debug=True)
    # else:
    #     print("Application could not start due to missing resources or Firebase initialization failure.")
