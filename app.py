from flask import Flask, request, jsonify
import pickle
import numpy as np
import zipfile
import os

app = Flask(__name__)

# --- NEW: AUTO-UNZIP LOGIC ---
# If the raw model file doesn't exist yet, unzip it!
if not os.path.exists('model.pkl'):
    print("Extracting model.zip...")
    with zipfile.ZipFile('model.zip', 'r') as zip_ref:
        zip_ref.extractall()
    print("Extraction Complete!")

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract features (Format: Limit, Sex, Edu, Mar, Age, Pay_Delay x 2, ... Bill)
    features = [0] * 23
    features[0] = data.get('limit', 50000)
    features[1] = 2
    features[2] = 2
    features[3] = 1
    features[4] = data.get('age', 25)
    features[5] = data.get('pay_delay', 0) # The most important input
    features[6] = data.get('pay_delay', 0)
    features[11] = data.get('bill_amt', 0)

    prediction = model.predict([features])
    
    # Send readable result
    result = "High Risk" if prediction[0] == 1 else "Safe"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
