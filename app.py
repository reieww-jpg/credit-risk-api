from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (We will upload the model file next)
# Note: You need to save your model in Colab first using pickle!
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract data
    features = [0] * 23
    features[0] = data.get('limit', 0)
    features[1] = 2  # Default Female
    features[2] = 2  # Default University
    features[3] = 1  # Default Married
    features[4] = data.get('age', 30)
    features[5] = data.get('pay_delay', 0)
    features[6] = data.get('pay_delay', 0)
    features[11] = data.get('bill_amt', 0)

    prediction = model.predict([features])
    result = "High Risk" if prediction[0] == 1 else "Safe"
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
