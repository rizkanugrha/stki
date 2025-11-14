# ==============================================================================
# == KODE FULL PERBAIKAN UNTUK: UTS/app/main.py
# ==============================================================================

import streamlit as st
import os
import sys
import pandas as pd
import nltk  # 1. Import NLTK

# --- PERBAIKAN DEPLOYMENT STREAMLIT (Soal LookupError) ---
# Menambahkan 3 baris ini untuk mengunduh data NLTK
# Ini akan berjalan saat server Streamlit membangun (build) aplikasi Anda
print("Memulai download data NLTK (stopwords & punkt)...")
nltk.download('stopwords')
nltk.download('punkt')
print("Download NLTK selesai.")
# --------------------------------------------------------


# --- PERBAIKAN PATH IMPORT (Soal 'src' not found) ---
# Tambahkan path root (UTS) ke sys.path agar bisa import 'src'
# Ini penting karena app/main.py ada di dalam subfolder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --------------------------------------------------------

# Impor modul Anda SEKARANG (setelah NLTK siap)
from src.vsm_ir import VectorSpaceModel
from src.boolean_ir import BooleanRetrieval
from src.preprocess import preprocess

# --- PERBAIKAN PATH DATA (Soal 'data/raw' not found) ---
# Path harus relatif dari root proyek (UTS), bukan dari file app/main.py
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
# --------------------------------------------------------

@st.cache_resource
def load_models():
    """
    Memuat model VSM & Boolean.
    Akan otomatis menjalankan preprocessing jika data/processed kosong.
    """
    # Pastikan data yang diproses ada
    # [PERBAIKAN] Menggunakan path yang sudah benar
    if not os.path.exists(PROCESSED_DATA_DIR) or not os.listdir(PROCESSED_DATA_DIR):
        st.warning(f"Data yang diproses ('{PROCESSED_DATA_DIR}') tidak ditemukan. Menjalankan preprocessing...")
        
        # Pastikan data mentah ada
        # [PERBAIKAN] Menggunakan path yang sudah benar
        if not os.path.exists(RAW_DATA_DIR):
            st.error(f"FATAL: Folder {RAW_DATA_DIR} tidak ditemukan. Tidak bisa memuat data.")
            return None, None
            
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.txt')]
        
        if not raw_files:
            st.error(f"Tidak ada file .txt di {RAW_DATA_DIR}.")
            return None, None
            
        # Lakukan preprocessing
        for doc_id in raw_files:
            with open(os.path.join(RAW_DATA_DIR, doc_id), 'r', encoding='utf-8') as f_in:
                raw_content = f_in.read()
            processed_content = preprocess(raw_content)
            with open(os.path.join(PROCESSED_DATA_DIR, doc_id), 'w', encoding='utf-8') as f_out:
                f_out.write(processed_content)
        st.success("Preprocessing selesai. Model siap dimuat.")

    # Muat model
    # [PERBAIKAN] Menggunakan path yang sudah benar
    vsm_model = VectorSpaceModel(RAW_DATA_DIR, sublinear_tf=True) 
    bool_model = BooleanRetrieval(PROCESSED_DATA_DIR)
    return vsm_model, bool_model

# --- Tampilan Utama Aplikasi Streamlit ---
st.title("Mesin Pencari Berita COVID-19 (UTS STKI)")
st.write("Project ini mengimplementasikan Boolean Retrieval dan Vector Space Model.")

try:
    vsm_model, bool_model = load_models()

    if vsm_model and bool_model:
        # --- Bagian Vector Space Model (VSM) - [Soal 04 & 05] ---
        st.header("Pencarian Peringkat (Vector Space Model)")
        vsm_query = st.text_input("Masukkan query VSM (misal: 'varian delta jakarta'):", key="vsm_query")
        k_value = st.slider("Jumlah hasil (Top-K):", min_value=1, max_value=5, value=3)
        
        if st.button("Cari (VSM)"):
            if vsm_query:
                results = vsm_model.search(vsm_query, k=k_value)
                st.subheader(f"Top {k_value} Hasil Pencarian VSM:")
                
                if not results:
                    st.warning("Tidak ada dokumen relevan yang ditemukan (skor > 0.01).")
                else:
                    for res in results:
                        st.markdown(f"**Dokumen:** `{res['doc_id']}` | **Skor:** `{res['score']:.4f}`")
                        st.info(f"**Snippet:** {res['snippet']}")
                        with st.expander("Lihat Teks Asli"):
                            doc_index = vsm_model.doc_ids.index(res['doc_id'])
                            st.text(vsm_model.raw_documents[doc_index])
            else:
                st.error("Silakan masukkan query VSM.")

        st.divider()

        # --- Bagian Boolean Retrieval - [Soal 03] ---
        st.header("Pencarian Boolean")
        st.info("Gunakan operator AND, OR, NOT. (Contoh: 'vaksin AND delta', 'ppkm OR jakarta')")
        bool_query = st.text_input("Masukkan query Boolean:", key="bool_query")
        
        if st.button("Cari (Boolean)"):
            if bool_query:
                results, explain = bool_model.process_query(bool_query)
                st.subheader("Hasil Pencarian Boolean:")
                
                st.code(f"Penjelasan: {explain}", language="text")
                st.markdown(f"Ditemukan **{len(results)}** dokumen:")
                
                if not results:
                    st.warning("Tidak ada dokumen yang cocok dengan query.")
                else:
                    for doc_id in results:
                        st.success(f"`{doc_id}`")
                        with st.expander("Lihat Teks (Sudah Diproses)"):
                            st.text(bool_model.doc_contents[doc_id])
            else:
                st.error("Silakan masukkan query Boolean.")

except Exception as e:
    st.error(f"Terjadi kesalahan fatal saat memuat model: {e}")
    st.error("Pastikan file data 'data/raw' ada dan 'src' dapat diakses.")
    st.code(f"Detail error: {e}", language="text")
