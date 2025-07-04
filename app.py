from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

# Load the saved model and pre-trained objects
try:
    with open('weather_prediction_model.pkl', 'rb') as file:
        model_data = joblib.load(file)
    temp_model = model_data['model']
    le_condition = model_data['le_condition']
    le_wdir = model_data['le_wdir']
    le_clds = model_data['le_clds']
    le_daynight = model_data['le_daynight']
    print(f"Model type: {type(temp_model)}")  # Debug print
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
            'temp': float(data.get('temp', 0)),
            'humidity': float(data.get('humidity', 0)),
            'windspeed': float(data.get('windspeed', 0)),
            'cloudcover': float(data.get('cloudcover', 0)),
            'sealevelpressure': float(data.get('sealevelpressure', 0)),
            'feelslike': float(data.get('feelslike', 0)),
            'dew': float(data.get('dew', 0)),
            'precipprob': float(data.get('precipprob', 0)),
            'uvindex': float(data.get('uvindex', 0)),
            'month': float(data.get('month', 6)),
            'hour': float(data.get('hour', 23)),
            'daynight': data.get('daynight', 'N'),
            'condition': data.get('condition', 'Haze'),
            'wdir_cardinal': data.get('wdir_cardinal', 'E'),
            'clds': data.get('clds', 'SCT')
        }

        # Encode categorical variables using the loaded label encoders
        input_data['condition_encoded'] = le_condition.transform([input_data['condition']])[0]
        input_data['wdir_cardinal_encoded'] = le_wdir.transform([input_data['wdir_cardinal']])[0]
        input_data['clds_encoded'] = le_clds.transform([input_data['clds']])[0]
        input_data['daynight_encoded'] = le_daynight.transform([input_data['daynight']])[0]

        # Prepare input DataFrame with model features
        features = ['month', 'hour', 'daynight_encoded', 'humidity %', 'pressure', 'visibility', 'wspd',
                   'condition_encoded', 'wdir_cardinal_encoded', 'clds_encoded']
        input_df = pd.DataFrame({
            'month': [input_data['month']],
            'hour': [input_data['hour']],
            'daynight_encoded': [input_data['daynight_encoded']],
            'humidity %': [input_data['humidity']],
            'pressure': [input_data['sealevelpressure']],
            'visibility': [10],
            'wspd': [input_data['windspeed']],
            'condition_encoded': [input_data['condition_encoded']],
            'wdir_cardinal_encoded': [input_data['wdir_cardinal_encoded']],
            'clds_encoded': [input_data['clds_encoded']]
        })

        print(f"Input DataFrame: {input_df}")

        # Predict temperature
        temp = temp_model.predict(input_df)[0]

        # Return JSON response
        return jsonify({
            'temperature': round(temp, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)