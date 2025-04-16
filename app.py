import joblib
import pandas as pd




def predict_fraud(data_path, model_path='model.pkl', scaler_path='scaler.pkl'):
    try:
       

        # Load the model and scaler
        model = joblib.load(model_path)
        model_features = model.get_booster().feature_names
        scaler = joblib.load(scaler_path)

        # Read data from CSV and take the first row
        input_df = pd.read_csv(data_path)
        input_df = input_df.iloc[[0]]  # Select only the first row for prediction

        print("Input DataFrame columns:", input_df.columns)

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
        categorical_features = ['gender', 'state','category', 'merchant']
        numerical_features = ['zip', 'city_pop', 'unix_time', 'amt']

        for col in categorical_features:
            input_df[col] = input_df[col].cat.codes

        for col in ['dob', 'trans_date', 'trans_time']:
            input_df[col] = input_df[col].astype('int64') / 10**9

        # Scale numerical features
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Align features with model input
        print("Final input features:", input_df.columns)
        print("Model expects:", model_features)
        input_df = input_df[model_features]

        # Predict
        prediction = model.predict(input_df)
        return int(prediction[0])

    except Exception as e:
        print(f"Error during prediction: {e}")
        return -1



from flask import Flask, request, jsonify
import os
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the fraud prediction service!"


@app.route('/predict', methods=['POST'])
def predict():
    #data_path='/content/drive/MyDrive/AI/dataset/diploma/X_val.csv'
    file = request.files['file']
    file_path = f'/tmp/{file.filename}'
    file.save(file_path)
    prediction = predict_fraud(file_path)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
