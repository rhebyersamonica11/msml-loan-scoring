import dagshub
import mlflow
import mlflow.sklearn
import os
import sys

# Tambahkan path agar bisa import modelling
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modelling import prepare_data, train_baseline_model, evaluate_model
from sklearn.ensemble import RandomForestClassifier

# Inisialisasi DagsHub
dagshub.init(repo_owner='rhebyersamonica11', repo_name='msml-loan-scoring', mlflow=True)

def run_experiment():
    data_path = os.path.join(os.path.dirname(__file__), "namadataset_preprocessing", "loan_clean.csv")
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    
    mlflow.set_experiment("Loan_Approval_Classification")
    
    # 1. Baseline (Akan ter-autolog karena fungsi dipanggil dari modelling.py)
    with mlflow.start_run(run_name="Baseline_Logistic_Regression"):
        model = train_baseline_model(X_train, y_train)
        print("Baseline Model training with autologging.")

    # 2. Tuning RF (Akan ter-autolog juga)
    configs = [{"name": "RF_Tuning_1", "n": 50, "depth": 5}, {"name": "RF_Tuning_2", "n": 150, "depth": 12}]
    for cfg in configs:
        with mlflow.start_run(run_name=cfg["name"]):
            rf = RandomForestClassifier(n_estimators=cfg["n"], max_depth=cfg["depth"], random_state=42)
            rf.fit(X_train, y_train)
            print(f"[+] {cfg['name']} Selesai (Autologged).")

if __name__ == "__main__":
    run_experiment()