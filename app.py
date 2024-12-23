from flask import Flask, render_template, request, url_for
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='.')

# Load model and scaler
model = joblib.load(os.path.join(os.path.dirname(__file__), 'cancer_model.pkl'))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'scaler.pkl'))

@app.before_request
def log_request():
    print(f"Request path: {request.path}")

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction_text = ""
    
    if request.method == "POST":
        # Retrieve input data
        nama_pasien = request.form['Nama Pasien']
        usia = float(request.form['Usia'])
        area_tumor = float(request.form['Area Tumor (mm²)'])
        keliling_tumor = float(request.form['Keliling Tumor (mm)'])
        radius_tumor = float(request.form['Radius Tumor (mm)'])
        poin_konkaf = float(request.form['Poin Konkaf Terburuk (mm)'])

        # Prepare input data for prediction
        input_data = np.array([area_tumor, keliling_tumor, radius_tumor, poin_konkaf])
        input_data_scaled = scaler.transform(input_data.reshape(1, -1))

        # Make prediction
        prediction = model.predict(input_data_scaled)

        if prediction == 1:
            prediction_text = "Malignant"  # Kanker ganas
        else:
            prediction_text = "Benign"  # Kanker jinak

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
