import os
import pandas as pd
import numpy as np

def load_raw_data(file_path):
    """Memuat data mentah dari jalur berkas yang ditentukan."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[-] Berkas tidak ditemukan di: {file_path}")
    print(f"[1/4] Memuat data mentah dari {file_path}...")
    return pd.read_csv(file_path)

def clean_missing_and_duplicates(data_frame):
    """Menangani baris duplikat dan mengisi nilai yang hilang (missing values)."""
    print("[2/4] Menjalankan pembersihan data duplikat dan missing values...")
    df = data_frame.copy()
    
    # Menghapus data duplikat
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        print("    [+] Data duplikat berhasil dihapus.")
        
    # Mengisi missing values numerik dengan median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Mengisi missing values kategorikal dengan modus
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    return df

def transform_features(data_frame):
    """Melakukan rekayasa fitur menggunakan One-Hot Encoding."""
    print("[3/4] Melakukan encoding pada fitur-fitur kategorikal...")
    df = data_frame.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Transformasi variabel kategori ke bentuk dummy binary
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Memastikan output berupa numerik (0/1), bukan boolean (True/False)
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
            
    return df_encoded

def save_clean_data(data_frame, output_directory, file_name="clean_data.csv"):
    """Menyimpan data hasil pemrosesan ke dalam folder output."""
    os.makedirs(output_directory, exist_ok=True)
    final_path = os.path.join(output_directory, file_name)
    data_frame.to_csv(final_path, index=False)
    print(f"[4/4] Sukses! Data bersih berhasil disimpan di: {final_path}")

def run_automation_pipeline():
    """Menjalankan seluruh alur pipa otomasi preprocessing."""
    # Definisikan jalur relatif sesuai struktur folder lokal Anda
    INPUT_FILE = "../namadataset_raw/raw_data.csv"
    OUTPUT_DIR = "namadataset_preprocessing"
    
    print("=== MEMULAI AUTOMATION PIPELINE PREPROCESSING ===")
    try:
        raw_df = load_raw_data(INPUT_FILE)
        cleaned_df = clean_missing_and_duplicates(raw_df)
        final_df = transform_features(cleaned_df)
        save_clean_data(final_df, OUTPUT_DIR)
        print("=== PIPELINE SELESAI DIEKSEKUSI DENGAN SUKSES ===")
    except Exception as e:
        print(f"[-] Terjadi kesalahan saat menjalankan pipeline: {str(e)}")

if __name__ == "__main__":
    run_automation_pipeline()