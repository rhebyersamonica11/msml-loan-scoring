import time
import random
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Histogram

# 1. Inisialisasi Flask App
app = Flask(__name__)

# 2. Definisikan Metrik Prometheus sesuai Kriteria Monitoring
PREDICTION_COUNTER = Counter(
    'model_predictions_total', 
    'Total jumlah prediksi yang dilakukan oleh model',
    ['status']
)
PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds', 
    'Waktu yang dibutuhkan untuk melakukan inferensi model'
)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    try:
        # Mengambil data input JSON dari request
        data = request.get_json()
        
        # --- Simulasi Jalannya Inferensi Model ---
        # Catatan: Di dunia nyata, kita memuat model pickle/joblib di sini.
        # Untuk keperluan monitoring metric, kita simulasikan proses pembacaannya.
        time.sleep(random.uniform(0.05, 0.2))  # Simulasi latensi model
        
        # Anggap prediksi sukses dan menghasilkan keputusan loan (1 = disetujui, 0 = ditolak)
        prediction_result = random.choice([0, 1])
        
        # Catat metrik sukses ke Prometheus
        PREDICTION_COUNTER.labels(status='success').inc()
        
        # Hitung durasi latensi
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        
        return jsonify({
            'status': 'success',
            'prediction': prediction_result,
            'latency_seconds': latency
        }), 200

    except Exception as e:
        # Catat metrik error ke Prometheus jika input bermasalah
        PREDICTION_COUNTER.labels(status='error').inc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    # Menjalankan server metrik Prometheus pada port 8000
    print("Mulai Prometheus Exporter di http://localhost:8000/metrics")
    start_http_server(8000)
    
    # Menjalankan server aplikasi web inferensi (Flask) pada port 5000
    print("Mulai Model Serving API di http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)