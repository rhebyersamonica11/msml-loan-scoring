from flask import Flask, request, jsonify
import time
import psutil
from prometheus_client import start_http_server, Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# --- 10 METRIK  ---
REQUEST_COUNT = Counter('api_requests_total', 'Total HTTP requests')
LATENCY = Histogram('model_inference_seconds', 'Waktu inferensi model')
PRED_RESULTS = Counter('model_predictions_total', 'Hasil prediksi', ['label'])
ERRORS = Counter('api_errors_total', 'Total HTTP errors', ['code'])
NULL_INPUTS = Counter('model_input_null_count', 'Jumlah input kosong')
SYS_MEM = Gauge('system_memory_usage', 'RAM usage')
SYS_CPU = Gauge('system_cpu_usage', 'CPU usage')
MODEL_STATUS = Gauge('model_loaded', 'Status model')
DATA_VOL = Counter('data_volume_processed', 'Jumlah data diproses')
HTTP_DUR = Histogram('http_duration', 'Durasi HTTP')

MODEL_STATUS.set(1)

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    data = request.get_json()
    
    if not data:
        NULL_INPUTS.inc()
        return jsonify({'error': 'No data'}), 400
    
    # Simulasi prediksi (Ganti dengan model.predict Anda)
    pred = 1 
    
    # Update Metrik
    REQUEST_COUNT.inc()
    LATENCY.observe(time.time() - start)
    PRED_RESULTS.labels(label=str(pred)).inc()
    DATA_VOL.inc()
    
    return jsonify({'prediction': pred})

@app.route('/metrics')
def metrics():
    SYS_MEM.set(psutil.virtual_memory().used)
    SYS_CPU.set(psutil.cpu_percent())
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    start_http_server(8000) # Port untuk Prometheus
    app.run(port=5000)      # Port untuk API