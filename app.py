import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
import os

# Define the function to predict fraud
def predict_fraud(data_path, model_path='model.pkl', scaler_path='scaler.pkl'):
    try:
        # Load the model and scaler from the project directory
        model = joblib.load(model_path)
        model_features = model.get_booster().feature_names
        scaler = joblib.load(scaler_path)

        # Read data from the uploaded CSV file
        input_df = pd.read_csv(data_path)
        input_df = input_df.iloc[[0]]  # Select only the first row for prediction

        # Convert columns to categories
        input_df['gender'] = input_df['gender'].astype('category')
        input_df['state'] = input_df['state'].astype('category')
        input_df['category'] = input_df['category'].astype('category')
        input_df['merchant'] = input_df['merchant'].astype('category')

        # Convert to datetime
        input_df['trans_time'] = pd.to_datetime(input_df['trans_time'])
        input_df['trans_date'] = pd.to_datetime(input_df['trans_date'])
        input_df['dob'] = pd.to_datetime(input_df['dob'])

        # Feature engineering
        categorical_features = ['gender', 'state', 'category', 'merchant']
        numerical_features = ['zip', 'city_pop', 'unix_time', 'amt']

        for col in categorical_features:
            input_df[col] = input_df[col].cat.codes

        for col in ['dob', 'trans_date', 'trans_time']:
            input_df[col] = input_df[col].astype('int64') / 10**9

        # Scale numerical features
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Align features with model input
        input_df = input_df[model_features]

        # Predict using the model
        prediction = model.predict(input_df)
        return int(prediction[0])

    except Exception as e:
        print(f"Error during prediction: {e}")
        return -1

# Flask app setup
app = Flask(__name__)

# Home route to serve the HTML form
@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html is in the 'templates' folder

# Prediction route to handle file upload
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Save the file temporarily
        file_path = f'/tmp/{file.filename}'
        file.save(file_path)

        # Call the prediction function
        prediction = predict_fraud(file_path)

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
