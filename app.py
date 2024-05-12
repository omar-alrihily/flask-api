import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your LG_model2 only once
with open('LG_model2.pkl', 'rb') as model_file:
    LG_model2 = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input data
    # Ensure all required keys are present in the input
    required_keys = ['Type', 'Options', 'Mileage', 'vehicle_age']
    if not all(key in data for key in required_keys):
        return jsonify({'error': 'Please provide Type, Options, Mileage, and vehicle_age'})

    # Extract input features
    vehicle_type = data['Type']
    options = data['Options']
    mileage = data['Mileage']
    vehicle_age = data['vehicle_age']

    # Process the data using your LG_model2
    prediction = LG_model2.predict(pd.DataFrame([[vehicle_type, options, mileage, vehicle_age]],
                                                columns=['Type', 'Options', 'Mileage', 'vehicle_age']))

    # Round the prediction value to the nearest integer
    rounded_prediction = np.round(prediction)  # Rounding using NumPy

    return jsonify({'prediction': int(rounded_prediction[0])})  # Return rounded prediction as JSON

if __name__ == '__main__':
   app.run(host='0.0.0.0', debug=True)
