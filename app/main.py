# ==============================================================================
# == KODE FULL PERBAIKAN UNTUK: UTS/app/main.py
# == (Versi ini MENGGANTI TOTAL UI agar sesuai screenshot)
# ==============================================================================

import streamlit as st
import os
import sys
import pandas as pd
import nltk  # 1. Import NLTK

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Mini Search Engine (UTS)", # Judul tab di browser
    page_icon="🔎",
    layout="wide" # Gunakan layout lebar
)
# --------------------------------------------------------

# --- PERBAIKAN DEPLOYMENT STREAMLIT (Soal LookupError) ---
print("Memulai download data NLTK (stopwords & punkt)...")
nltk.download('stopwords')
nltk.download('punkt')
print("Download NLTK selesai.")
# --------------------------------------------------------

# --- PERBAIKAN PATH IMPORT (Soal 'src' not found) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --------------------------------------------------------

# Impor modul Anda SEKARANG (setelah NLTK siap)
from src.vsm_ir import VectorSpaceModel
from src.boolean_ir import BooleanRetrieval
from src.preprocess import preprocess
from src.eval import (
    calculate_precision_recall_f1, 
    mean_average_precision, 
    precision_at_k
)

# --- PERBAIKAN PATH DATA (Soal 'data/raw' not found) ---
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
# --------------------------------------------------------

# --- KUNCI JAWABAN (GOLD STANDARD) ---
# Kita gabungkan semua kunci jawaban di sini
GOLD_STANDARD_ALL = {
    # Kunci Jawaban VSM
    "vaksin AND delta": {"berita2.txt", "berita3.txt"},
    "ppkm OR jakarta": {"berita1.txt", "berita4.txt"},
    "kasus NOT amerika": {"berita1.txt", "berita4.txt"},
    
    # Kunci Jawaban Boolean (Bisa jadi beda, tapi kita samakan saja untuk VSM)
    # (Diambil dari evaluasi Soal 03 Anda)
    "vaksin AND delta": {"berita3.txt"},
    "ppkm OR jakarta": {"berita1.txt", "berita4.txt"},
    "kasus NOT amerika": {"berita1.txt", "berita2.txt", "berita3.txt", "berita4.txt"}
}
# --------------------------------------------------------


@st.cache_resource
def load_models():
    """
    Memuat semua model yang diperlukan saat startup.
    """
    if not os.path.exists(PROCESSED_DATA_DIR) or not os.listdir(PROCESSED_DATA_DIR):
        print(f"Data yang diproses ('{PROCESSED_DATA_DIR}') tidak ditemukan. Menjalankan preprocessing...")
        if not os.path.exists(RAW_DATA_DIR):
            print(f"FATAL: Folder {RAW_DATA_DIR} tidak ditemukan.")
            return None, None, None
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.txt')]
        if not raw_files:
            print(f"Tidak ada file .txt di {RAW_DATA_DIR}.")
            return None, None, None
        for doc_id in raw_files:
            with open(os.path.join(RAW_DATA_DIR, doc_id), 'r', encoding='utf-8') as f_in:
                raw_content = f_in.read()
            processed_content = preprocess(raw_content)
            with open(os.path.join(PROCESSED_DATA_DIR, doc_id), 'w', encoding='utf-8') as f_out:
                f_out.write(processed_content)
        print("Preprocessing selesai.")

    print("Memuat semua model...")
    vsm_model_default = VectorSpaceModel(RAW_DATA_DIR, sublinear_tf=False) 
    vsm_model_sublinear = VectorSpaceModel(RAW_DATA_DIR, sublinear_tf=True)
    bool_model = BooleanRetrieval(PROCESSED_DATA_DIR)
    print("Semua model berhasil dimuat!")
    return vsm_model_default, vsm_model_sublinear, bool_model

@st.cache_data
def load_raw_docs_to_dict(raw_dir):
    """Memuat dokumen mentah ke dictionary untuk lookup snippet."""
    docs_dict = {}
    if not os.path.exists(raw_dir):
        return docs_dict
    
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.txt')]
    for doc_id in raw_files:
        with open(os.path.join(raw_dir, doc_id), 'r', encoding='utf-8') as f:
            docs_dict[doc_id] = f.read()
    return docs_dict

# ==========================================================
# --- Tampilan Utama Aplikasi Streamlit (LAYOUT BARU) ---
# ==========================================================
st.title("Mesin Pencari Berita COVID-19 (UTS STKI)")
st.text(f"Nama: Rizka Nugraha NIM: A11.2022.14119")
st.write("Project ini mengimplementasikan Boolean Retrieval dan Vector Space Model.")

try:
    # Coba muat model
    vsm_model_default, vsm_model_sublinear, bool_model = load_models()
    raw_docs_dict = load_raw_docs_to_dict(RAW_DATA_DIR)

    if not vsm_model_default or not bool_model or not raw_docs_dict:
        st.error("Gagal memuat model atau data mentah. Periksa log.")
    else:
        # --- UI Input (Sesuai Screenshot) ---
        query = st.text_input("Masukkan query:")
        
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox(
                "Skema pencarian",
                ["VSM Sublinear", "VSM TF-IDF", "Boolean"]
            )
        with col2:
            k_value = st.slider("Top-k dokumen", 1, 5, 5) # Corpus kita kecil, 5 sudah cukup

        # --- Tombol "Cari" ---
        if st.button("Cari"):
            if not query:
                st.warning("Silakan masukkan query.")
            else:
                # Siapkan variabel untuk hasil
                st.subheader("Hasil Pencarian")
                retrieved_ids = []
                results_list_of_dicts = [] # Hanya untuk VSM, untuk MAP
                
                # --- Logika Pencarian ---
                if model_type == "Boolean":
                    retrieved_ids, explain = bool_model.process_query(query)
                    st.code(f"Penjelasan Kueri: {explain}")
                    if not retrieved_ids:
                        st.info("Tidak ada dokumen yang cocok.")
                    else:
                        # Tampilkan hasil boolean
                        for i, doc_id in enumerate(retrieved_ids):
                            c1, c2, c3 = st.columns([0.5, 1.5, 6])
                            c1.write(f"**{i}**")
                            c2.write(f"`{doc_id}`")
                            snippet = raw_docs_dict.get(doc_id, "Snippet tidak ditemukan.")
                            snippet = snippet.replace('\n', ' ').strip()[:100] + "..."
                            c3.write(snippet)

                else: # Jika VSM (Sublinear atau TF-IDF)
                    model_to_use = vsm_model_sublinear if model_type == "VSM Sublinear" else vsm_model_default
                    
                    results_list_of_dicts = model_to_use.search(query, k=k_value)
                    retrieved_ids = [r['doc_id'] for r in results_list_of_dicts]
                    
                    if not results_list_of_dicts:
                        st.info("Tidak ada dokumen yang relevan (skor > 0.01).")
                    else:
                        # Tampilkan hasil VSM
                        for i, res in enumerate(results_list_of_dicts):
                            c1, c2, c3, c4 = st.columns([0.5, 1.5, 1, 6])
                            c1.write(f"**{i}**")
                            c2.write(f"`{res['doc_id']}`")
                            c3.write(f"**{res['score']:.4f}**")
                            c4.write(res['snippet'])

                # --- Bagian Evaluasi (Real-time) ---
                st.subheader("Evaluasi")
                relevant_docs = GOLD_STANDARD_ALL.get(query)
                
                if relevant_docs:
                    st.success(f"Kunci jawaban (gold standard) ditemukan untuk kueri ini!")
                    
                    # Hitung P, R, F1 (berlaku untuk semua)
                    p, r, f1 = calculate_precision_recall_f1(retrieved_ids, relevant_docs)
                    
                    # Hitung MAP (hanya untuk VSM)
                    map5 = 0.0
                    if model_type != "Boolean":
                        # {query: results_list_of_dicts} -> Ini adalah query_results
                        # {query: relevant_docs} -> Ini adalah gold_standard
                        map5 = mean_average_precision({query: results_list_of_dicts}, {query: relevant_docs}, k=5)

                    # Tampilkan metrik (sesuai screenshot)
                    st.text(f"Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}, MAP@5: {map5:.2f}, nDCG@5: N/A")

                else:
                    st.info("Kunci jawaban (gold standard) tidak ditemukan untuk kueri ini. Evaluasi real-time tidak dapat dihitung.")

except Exception as e:
    # Tangkap error umum (seperti import gagal, dll)
    st.error(f"Terjadi kesalahan fatal saat aplikasi dimulai: {e}")
    st.error("Pastikan file data 'data/raw' ada dan 'src' dapat diakses.")
    st.code(f"Detail error: {e}", language="text")