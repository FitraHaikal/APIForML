import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS
import os
import json

# Create flask app
flask_app = Flask(__name__)
CORS(flask_app)  # Mengizinkan CORS untuk request dari frontend

# Load model dan transformer
try:
    model = pickle.load(open("linear_regression_model.pkl", "rb"))
    transformer = pickle.load(open("transformer.pkl", "rb"))
    print("✅ Model dan transformer berhasil dimuat!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    transformer = None

# Load dataset untuk ditampilkan di frontend
try:
    df = pd.read_csv("healthcare_dataset.csv")
    print(f"✅ Dataset berhasil dimuat! Total {len(df)} records")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    df = None


@flask_app.route("/")
def Home():
    """Halaman utama dengan form prediksi dan tampilan data"""
    return render_template("index.html")


@flask_app.route("/api/data")
def get_data():
    """Endpoint untuk mendapatkan data dataset (untuk ditampilkan di frontend)"""
    try:
        if df is None:
            return jsonify({'error': 'Dataset tidak tersedia', 'data': []}), 500
        
        # Ambil hanya kolom yang diperlukan untuk ditampilkan
        data = df[['Age', 'Gender', 'Blood Type', 'Medical Condition', 'Billing Amount']].head(500).to_dict('records')
        
        # Format ulang untuk konsistensi dengan frontend
        formatted_data = []
        for row in data:
            formatted_data.append({
                'Age': int(row['Age']),
                'Gender': row['Gender'],
                'BloodType': row['Blood Type'],
                'MedicalCondition': row['Medical Condition'],
                'BillingAmount': float(row['Billing Amount'])
            })
        
        # Hitung statistik tambahan
        billing_amounts = [row['BillingAmount'] for row in formatted_data]
        
        return jsonify({
            'data': formatted_data,
            'total': len(formatted_data),
            'statistics': {
                'avg_billing': np.mean(billing_amounts),
                'min_billing': np.min(billing_amounts),
                'max_billing': np.max(billing_amounts),
                'std_billing': np.std(billing_amounts)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e), 'data': []}), 500


@flask_app.route("/api/statistics")
def get_statistics():
    """Endpoint untuk mendapatkan statistik lengkap dataset"""
    try:
        if df is None:
            return jsonify({'error': 'Dataset tidak tersedia'}), 500
        
        # Statistik berdasarkan Kondisi Medis
        condition_stats = df.groupby('Medical Condition')['Billing Amount'].agg(['mean', 'count', 'min', 'max']).to_dict()
        
        # Statistik berdasarkan Gender
        gender_stats = df.groupby('Gender')['Billing Amount'].agg(['mean', 'count']).to_dict()
        
        # Statistik berdasarkan Golongan Darah
        blood_stats = df.groupby('Blood Type')['Billing Amount'].agg(['mean', 'count']).to_dict()
        
        # Statistik berdasarkan Umur (kelompok)
        age_bins = [0, 18, 30, 45, 60, 80, 120]
        age_labels = ['0-18', '19-30', '31-45', '46-60', '61-80', '80+']
        df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
        age_stats = df.groupby('Age Group')['Billing Amount'].agg(['mean', 'count']).to_dict()
        
        return jsonify({
            'total_records': len(df),
            'unique_conditions': df['Medical Condition'].nunique(),
            'unique_blood_types': df['Blood Type'].nunique(),
            'condition_stats': {
                k: {'mean': v['mean'], 'count': v['count'], 'min': v['min'], 'max': v['max']} 
                for k, v in condition_stats.items()
            },
            'gender_stats': {
                k: {'mean': v['mean'], 'count': v['count']} 
                for k, v in gender_stats.items()
            },
            'blood_stats': {
                k: {'mean': v['mean'], 'count': v['count']} 
                for k, v in blood_stats.items()
            },
            'age_stats': {
                k: {'mean': v['mean'], 'count': v['count']} 
                for k, v in age_stats.items()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@flask_app.route("/predict", methods=["POST"])
def predict():
    """Endpoint untuk melakukan prediksi billing amount"""
    try:
        # Cek apakah model dan transformer tersedia
        if model is None or transformer is None:
            return render_template("index.html", error_text="Model belum dimuat. Silakan coba lagi nanti.")
        
        # Ambil data dari form (jika dari HTML form biasa)
        if request.form:
            age = float(request.form.get('Age'))
            gender = request.form.get('Gender')
            blood_type = request.form.get('Blood Type')
            medical_condition = request.form.get('Medical Condition')
        
        # Atau dari JSON (jika dari JavaScript fetch)
        elif request.is_json:
            data = request.get_json()
            age = float(data.get('Age'))
            gender = data.get('Gender')
            blood_type = data.get('Blood Type')
            medical_condition = data.get('Medical Condition')
        
        else:
            return jsonify({'error': 'Format request tidak didukung'}), 400
        
        # Validasi input
        if not all([age, gender, blood_type, medical_condition]):
            error_msg = "Semua field harus diisi!"
            if request.form:
                return render_template("index.html", error_text=error_msg)
            return jsonify({'error': error_msg}), 400
        
        if age < 0 or age > 120:
            error_msg = "Umur harus antara 0 - 120 tahun!"
            if request.form:
                return render_template("index.html", error_text=error_msg)
            return jsonify({'error': error_msg}), 400
        
        # Validasi nilai yang valid untuk kategorikal
        valid_genders = ['Male', 'Female']
        valid_blood_types = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        valid_conditions = ['Cancer', 'Diabetes', 'Hypertension', 'Asthma', 'Obesity', 'Arthritis']
        
        if gender not in valid_genders:
            error_msg = f"Jenis kelamin tidak valid. Pilih: {', '.join(valid_genders)}"
            if request.form:
                return render_template("index.html", error_text=error_msg)
            return jsonify({'error': error_msg}), 400
        
        if blood_type not in valid_blood_types:
            error_msg = f"Golongan darah tidak valid. Pilih: {', '.join(valid_blood_types)}"
            if request.form:
                return render_template("index.html", error_text=error_msg)
            return jsonify({'error': error_msg}), 400
        
        if medical_condition not in valid_conditions:
            error_msg = f"Kondisi medis tidak valid. Pilih: {', '.join(valid_conditions)}"
            if request.form:
                return render_template("index.html", error_text=error_msg)
            return jsonify({'error': error_msg}), 400
        
        # Buat DataFrame untuk input (HARUS SAMA PERSIS dengan training)
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Blood Type': blood_type,
            'Medical Condition': medical_condition
        }])
        
        # Transformasi fitur menggunakan one-hot encoding
        transformed_features = transformer.transform(input_df)
        
        # Prediksi
        prediction = model.predict(transformed_features)[0]
        
        # Format hasil ke Rupiah (atau USD)
        formatted_prediction = f"${prediction:,.2f}"
        
        # Siapkan response
        result = {
            'prediction': float(prediction),
            'billing_amount': float(prediction),
            'formatted_amount': formatted_prediction,
            'status': 'success',
            'message': f'Prediksi berhasil! Estimasi biaya: {formatted_prediction}'
        }
        
        # Jika request dari form (browser biasa), return template
        if request.form:
            return render_template("index.html", 
                                 prediction_text=formatted_prediction,
                                 predicted_value=prediction)
        
        # Jika request dari AJAX/JavaScript, return JSON
        return jsonify(result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error in prediction: {error_msg}")
        
        if request.is_json:
            return jsonify({'error': error_msg}), 400
        
        return render_template("index.html", error_text=f"Error: {error_msg}")


@flask_app.route("/api/predict", methods=["POST"])
def predict_api():
    """Endpoint API murni untuk prediksi (return JSON saja)"""
    try:
        if model is None or transformer is None:
            return jsonify({'error': 'Model belum dimuat'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        age = float(data.get('Age'))
        gender = data.get('Gender')
        blood_type = data.get('Blood Type')
        medical_condition = data.get('Medical Condition')
        
        # Validasi
        if not all([age, gender, blood_type, medical_condition]):
            return jsonify({'error': 'Semua field harus diisi'}), 400
        
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
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'billing_amount': float(prediction),
            'formatted_amount': f"${prediction:,.2f}",
            'input': data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@flask_app.route("/api/model-info")
def model_info():
    """Endpoint untuk mendapatkan informasi tentang model"""
    try:
        if model is None:
            return jsonify({'error': 'Model belum dimuat'}), 500
        
        # Dapatkan informasi model
        model_info = {
            'model_type': type(model).__name__,
            'has_transformer': transformer is not None,
            'features_used': ['Age', 'Gender', 'Blood Type', 'Medical Condition'],
            'target': 'Billing Amount',
            'transformer_type': type(transformer).__name__ if transformer else None
        }
        
        # Coba dapatkan koefisien jika tersedia
        if hasattr(model, 'coef_'):
            model_info['has_coefficients'] = True
            model_info['intercept'] = float(model.intercept_) if hasattr(model, 'intercept_') else None
        
        return jsonify(model_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Server berjalan di http://localhost:{port}")
    print(f"📊 Kunjungi http://localhost:{port} untuk membuka aplikasi\n")
    flask_app.run(debug=True, host='0.0.0.0', port=port)
