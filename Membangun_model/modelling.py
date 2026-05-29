import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def prepare_data(data_path):
    """Membaca data bersih dan memisahkan fitur dengan target."""
    df = pd.read_csv(data_path)
    
    # Menentukan target (loan_status) dan fitur
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # Split data 80% training dan 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def train_baseline_model(X_train, y_train):
    """Melatih model baseline menggunakan Logistic Regression."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Menghitung metrik evaluasi model."""
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    return acc, prec, rec