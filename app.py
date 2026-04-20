import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS
import os

# Create flask app
flask_app = Flask(__name__)
CORS(flask_app)

# Load model dan transformer (INI PENTING!)
model = pickle.load(open("linear_regression_model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data dari form
        age = float(request.form.get('Age'))
        gender = request.form.get('Gender')
        blood_type = request.form.get('Blood Type')
        medical_condition = request.form.get('Medical Condition')
        
        # Validasi input
        if not all([age, gender, blood_type, medical_condition]):
            return render_template("index.html", error_text="Semua field harus diisi!")
        
        if age < 0 or age > 120:
            return render_template("index.html", error_text="Umur harus antara 0-120 tahun!")
        
        # Buat DataFrame untuk input (HARUS SAMA PERSIS dengan training)
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Blood Type': blood_type,
            'Medical Condition': medical_condition
        }])
        
        # Transformasi fitur menggunakan transformer
        transformed_features = transformer.transform(input_df)
        
        # Prediksi
        prediction = model.predict(transformed_features)[0]
        
        # Format hasil ke USD
        formatted_prediction = f"${prediction:,.2f}"
        
        return render_template("index.html", 
                             prediction_text=formatted_prediction,
                             predicted_value=prediction)
        
    except Exception as e:
        return render_template("index.html", error_text=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    flask_app.run(debug=True, host='0.0.0.0', port=port)