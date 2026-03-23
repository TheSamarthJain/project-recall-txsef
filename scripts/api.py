from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, tempfile, os
import numpy as np

# Adjust import path so we can import from the same directory
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from advanced_feature_extraction import extract_advanced_features

app = Flask(__name__)
# Enable CORS for all routes so the frontend can call the API from any domain
CORS(app)

# Determine path to models directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load saved model + scaler (using logistic regression which got 100% in tests)
try:
    with open(os.path.join(MODELS_DIR, 'final_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    model = None

try:
    with open(os.path.join(MODELS_DIR, 'final_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading scaler: {e}")
    scaler = None

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint for Render to verify the app is up"""
    status = 'healthy' if model is not None and scaler is not None else 'unhealthy'
    return jsonify({
        'status': status,
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Server configuration error (model not loaded)'}), 500

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided. Please send a file in the "audio" field.'}), 400

    audio_file = request.files['audio']
    
    # Save temporarily to parse with librosa
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio_file.save(tmp.name)
        try:
            features = extract_advanced_features(tmp.name)
        except Exception as e:
            features = None
            print(f"Feature extraction error: {e}")
        finally:
            os.unlink(tmp.name)

    if features is None:
        return jsonify({'error': 'Feature extraction failed. Ensure the file is a valid 16kHz WAV or MP3.'}), 500

    # Clean up non-numeric metadata
    for key in ['filename', 'label', 'participant_id', 'corpus']:
        features.pop(key, None)

    # Convert to array (making sure order matches extraction script output)
    try:
        X = np.array(list(features.values())).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = scaler.transform(X)

        # Predict
        prob = model.predict_proba(X_scaled)[0][1]
        prediction = 'AD' if prob > 0.5 else 'HC'

        return jsonify({
            'prediction': prediction,
            'confidence': round(float(prob if prediction == 'AD' else 1 - prob), 3),
            'ad_probability': round(float(prob), 3)
        })
    except Exception as e:
        return jsonify({'error': f'Prediction logic failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Use port 5000 for local development, or process.env.PORT for Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
