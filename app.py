from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

# Load models and pre-trained objects
try:
    temp_model = joblib.load('temperature_model.pkl')
    condition_model = joblib.load('condition_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError as e:
    print(f"Error: {e}")
    raise

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form
        input_data = {
            'temp': float(data['temp']),
            'humidity': float(data['humidity']),
            'windspeed': float(data['windspeed']),
            'cloudcover': float(data['cloudcover']),
            'sealevelpressure': float(data['sealevelpressure']),
            'feelslike': float(data['feelslike']),
            'dew': float(data['dew']),
            'precipprob': float(data['precipprob']),
            'uvindex': float(data['uvindex'])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        print(f"Scaled Input: {input_scaled}")

        # Predict temperature and condition
        temp = temp_model.predict(input_scaled)[0]
        condition_encoded = condition_model.predict(input_scaled)[0]
        condition = le.inverse_transform([condition_encoded])[0]

        # Return JSON response
        return jsonify({
            'temperature': round(temp, 2),
            'condition': condition
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)