import time
import requests
import random

url = 'http://localhost:5000/predict'

# Contoh data simulasi yang dikirimkan ke model serving
payloads = [
    {"Gender": "Male", "Married": "Yes", "Dependents": "3+", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 5000, "CoapplicantIncome": 0, "LoanAmount": 150, "Loan_Amount_Term": 360, "Credit_History": 1.0, "Property_Area": "Urban"},
    {"Gender": "Female", "Married": "No", "Dependents": "0", "Education": "Graduate", "Self_Employed": "Yes", "ApplicantIncome": 4500, "CoapplicantIncome": 1800, "LoanAmount": 120, "Loan_Amount_Term": 360, "Credit_History": 1.0, "Property_Area": "Semiurban"},
    {"Gender": "Male", "Married": "Yes", "Dependents": "1", "Education": "Not Graduate", "Self_Employed": "No", "ApplicantIncome": 2500, "CoapplicantIncome": 1500, "LoanAmount": 80, "Loan_Amount_Term": 180, "Credit_History": 0.0, "Property_Area": "Rural"}
]

print("Memulai pengiriman data inferensi otomatis... (Tekan Ctrl+C untuk berhenti)")

try:
    while True:
        # Pilih payload secara acak
        data = random.choice(payloads)
        
        try:
            response = requests.post(url, json=data)
            print(f"Status Transmisi: {response.status_code} | Respons Model: {response.json()}")
        except Exception as e:
            print(f"Gagal terhubung ke Server Serving API: {e}")
            
        # Beri jeda acak antara 1 hingga 3 detik per request
        time.sleep(random.uniform(1, 3))
except KeyboardInterrupt:
    print("\nPengiriman data otomatis dihentikan.")