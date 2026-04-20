import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS
import os
import sys

# Create flask app
flask_app = Flask(__name__)
CORS(flask_app)

# Print untuk debugging di log Render
print("=" * 50)
print("🚀 Starting Healthcare Prediction API...")
print("=" * 50)

# Load model dan transformer dengan error handling
try:
    print("📂 Loading model.pkl...")
    model = pickle.load(open("linear_regression_model.pkl", "rb"))
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

try:
    print("📂 Loading transformer.pkl...")
    transformer = pickle.load(open("transformer.pkl", "rb"))
    print("✅ Transformer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading transformer: {e}")
    transformer = None

# Load dataset
try:
    print("📂 Loading healthcare_dataset.csv...")
    df = pd.read_csv("healthcare_dataset.csv")
    print(f"✅ Dataset loaded! Total {len(df)} records")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    df = None

print("=" * 50)


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/api/data")
def get_data():
    try:
        if df is None:
            return jsonify({'error': 'Dataset not available', 'data': []}), 500
        
        data = df[['Age', 'Gender', 'Blood Type', 'Medical Condition', 'Billing Amount']].head(500).to_dict('records')
        formatted_data = []
        for row in data:
            formatted_data.append({
                'Age': int(row['Age']),
                'Gender': row['Gender'],
                'BloodType': row['Blood Type'],
                'MedicalCondition': row['Medical Condition'],
                'BillingAmount': float(row['Billing Amount'])
            })
        
        return jsonify({'data': formatted_data, 'total': len(formatted_data)})
    except Exception as e:
        return jsonify({'error': str(e), 'data': []}), 500


@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or transformer is None:
            return render_template("index.html", error_text="Model not loaded. Please try again later.")
        
        if request.form:
            age = float(request.form.get('Age'))
            gender = request.form.get('Gender')
            blood_type = request.form.get('Blood Type')
            medical_condition = request.form.get('Medical Condition')
        elif request.is_json:
            data = request.get_json()
            age = float(data.get('Age'))
            gender = data.get('Gender')
            blood_type = data.get('Blood Type')
            medical_condition = data.get('Medical Condition')
        else:
            return jsonify({'error': 'Format tidak didukung'}), 400
        
        # Validasi
        if not all([age, gender, blood_type, medical_condition]):
            error_msg = "Semua field harus diisi!"
            if request.form:
                return render_template("index.html", error_text=error_msg)
            return jsonify({'error': error_msg}), 400
        
        if age < 0 or age > 120:
            error_msg = "Umur harus antara 0-120 tahun!"
            if request.form:
                return render_template("index.html", error_text=error_msg)
            return jsonify({'error': error_msg}), 400
        
        # Buat DataFrame
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Blood Type': blood_type,
            'Medical Condition': medical_condition
        }])
        
        # Transformasi dan prediksi
        transformed_features = transformer.transform(input_df)
        prediction = model.predict(transformed_features)[0]
        formatted_prediction = f"${prediction:,.2f}"
        
        if request.form:
            return render_template("index.html", 
                                 prediction_text=formatted_prediction,
                                 predicted_value=prediction)
        
        return jsonify({
            'prediction': float(prediction),
            'billing_amount': float(prediction),
            'formatted_amount': formatted_prediction,
            'status': 'success'
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error: {error_msg}")
        if request.is_json:
            return jsonify({'error': error_msg}), 400
        return render_template("index.html", error_text=f"Error: {error_msg}")


@flask_app.route("/health")
def health_check():
    """Endpoint untuk health check di Render"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'transformer_loaded': transformer is not None,
        'dataset_loaded': df is not None
    })


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Server running on http://localhost:{port}")
    print(f"📊 Health check: http://localhost:{port}/health\n")
    flask_app.run(debug=False, host='0.0.0.0', port=port)
