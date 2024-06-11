import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

from preprocess import process_list

app = Flask(__name__)

# Load the trained model
model = load_model('bank_churn_dnn.h5')

@app.route('/')
def home():
    return "Welcome to the Churn Prediction API!"

# @app.route('/predict', methods=['POST','GET'])
# def predict_file(): 
#     class_names = ["Churn","No Churn"]       
    
#     # Process the image and make predictions
#     data = request.get_json()
    
#     my_list = data['data']
    
#     my_list = process_list(my_list)
    
#     y_pred = (model.predict(my_list) > 0.5).astype("int32")
#     return jsonify({'predicted_class': y_pred})


@app.route('/predict', methods=['POST', 'GET'])
def predict_file():
    try:
        # Define class names
        class_names = ["Churn", "No Churn"]

        # Check if request is JSON
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        # Get JSON data
        data = request.get_json()

        # Check if data is provided
        if 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400

        # Process the list using the preprocess function
        my_list = data['data']
        processed_list = process_list(my_list)

        # Make prediction
        y_pred = (model.predict(processed_list) > 0.5).astype("int32")
        predicted_classes = [class_names[int(pred)] for pred in y_pred]

        return jsonify({'predicted_class': predicted_classes})

    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)
