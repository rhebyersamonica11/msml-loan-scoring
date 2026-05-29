import dagshub
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from modelling import prepare_data, train_baseline_model, evaluate_model

# 1. Inisialisasi koneksi remote otomatis ke DagsHub sesuai akun baru Anda
dagshub.init(
    repo_owner='rhebyersamonica11', 
    repo_name='msml-loan-scoring', 
    mlflow=True
)

def run_experiment():
    # Path data bersih di dalam folder Membangun_model
    DATA_PATH = "namadataset_preprocessing/clean_data.csv"
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
    
    # Set nama eksperimen di MLflow
    mlflow.set_experiment("Loan_Approval_Classification")
    
    # --- EXPERIMENT 1: BASELINE MODEL (Logistic Regression) ---
    with mlflow.start_run(run_name="Baseline_Logistic_Regression"):
        print("[+] Menjalankan Eksperimen 1: Baseline Model...")
        baseline_model = train_baseline_model(X_train, y_train)
        acc, prec, rec = evaluate_model(baseline_model, X_test, y_test)
        
        # Log parameter dan metrik ke DagsHub
        mlflow.log_param("model_type", "Logistic_Regression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        # Simpan model ke MLflow Artifacts
        mlflow.sklearn.log_model(baseline_model, "baseline_model")
        print(f"    Hasil -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")

    # --- EXPERIMENT 2: TUNING MODEL (Random Forest Variant 1) ---
    with mlflow.start_run(run_name="Random_Forest_Tuning_1"):
        print("[+] Menjalankan Eksperimen 2: Random Forest Variant 1...")
        n_estimators = 50
        max_depth = 5
        
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_model.fit(X_train, y_train)
        acc, prec, rec = evaluate_model(rf_model, X_test, y_test)
        
        # Log parameter dan metrik ke DagsHub
        mlflow.log_param("model_type", "Random_Forest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        mlflow.sklearn.log_model(rf_model, "rf_tuning_1_model")
        print(f"    Hasil -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")

    # --- EXPERIMENT 3: TUNING MODEL (Random Forest Variant 2 - Lebih Kompleks) ---
    with mlflow.start_run(run_name="Random_Forest_Tuning_2"):
        print("[+] Menjalankan Eksperimen 3: Random Forest Variant 2...")
        n_estimators = 150
        max_depth = 12
        
        rf_model2 = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_model2.fit(X_train, y_train)
        acc, prec, rec = evaluate_model(rf_model2, X_test, y_test)
        
        # Log parameter dan metrik ke DagsHub
        mlflow.log_param("model_type", "Random_Forest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        # Ini akan menjadi model terbaik kita yang akan di-register nanti
        mlflow.sklearn.log_model(rf_model2, "rf_tuning_2_model")
        print(f"    Hasil -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")

if __name__ == "__main__":
    run_experiment()