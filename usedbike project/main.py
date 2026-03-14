from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

app = Flask(__name__)

# Load and prepare data
def load_and_train_model():
    df = pd.read_csv('Used_Bikes.csv')
    
    # Prepare features (similar to notebook processing)
    df['model'] = df['age']  # Using age as model feature
    
    # Encode categorical variables
    le_city = LabelEncoder()
    le_owner = LabelEncoder()
    le_brand = LabelEncoder()
    
    df['city_encoded'] = le_city.fit_transform(df['city'])
    df['owner_encoded'] = le_owner.fit_transform(df['owner'])
    df['brand_encoded'] = le_brand.fit_transform(df['brand'])
    
    # Features for prediction
    features = ['kms_driven', 'owner_encoded', 'model', 'power', 'brand_encoded', 'city_encoded']
    X = df[features]
    y = df['price']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save encoders and model
    with open('model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'le_city': le_city,
            'le_owner': le_owner,
            'le_brand': le_brand
        }, f)
    
    return model, le_city, le_owner, le_brand

# Load model if exists, otherwise train new one
if os.path.exists('model.pkl'):
    with open('model.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
        le_city = saved_data['le_city']
        le_owner = saved_data['le_owner']
        le_brand = saved_data['le_brand']
else:
    model, le_city, le_owner, le_brand = load_and_train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        data = request.json
        
        # Get form data
        kms_driven = float(data['kms_driven'])
        owner = data['owner']
        age = float(data['age'])
        power = float(data['power'])
        brand = data['brand']
        city = data['city']
        
        # Encode categorical variables
        try:
            city_encoded = le_city.transform([city])[0]
        except ValueError:
            city_encoded = 0  # Default for unknown cities
            
        try:
            owner_encoded = le_owner.transform([owner])[0]
        except ValueError:
            owner_encoded = 0  # Default for unknown owner types
            
        try:
            brand_encoded = le_brand.transform([brand])[0]
        except ValueError:
            brand_encoded = 0  # Default for unknown brands
        
        # Prepare input features
        features = np.array([[kms_driven, owner_encoded, age, power, brand_encoded, city_encoded]])
        
        # Make prediction
        predicted_price = model.predict(features)[0]
        
        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'price_range': {
                'min': round(predicted_price * 0.9, 2),
                'max': round(predicted_price * 1.1, 2)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_options', methods=['GET'])
def get_options():
    df = pd.read_csv('Used_Bikes.csv')
    
    options = {
        'cities': sorted(df['city'].unique().tolist()),
        'brands': sorted(df['brand'].unique().tolist()),
        'owners': sorted(df['owner'].unique().tolist()),
        'power_options': sorted(df['power'].unique().tolist())
    }
    
    return jsonify(options)

if __name__ == '__main__':
    app.run(debug=True)