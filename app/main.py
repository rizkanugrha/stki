import streamlit as st
import os
import sys
import pandas as pd

# Tambahkan src ke path agar bisa import modul
# Ini penting karena app/main.py ada di folder berbeda
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vsm_ir import VectorSpaceModel
from src.boolean_ir import BooleanRetrieval
from src.preprocess import preprocess

# Path ke data (relatif dari root proyek 'UTS')
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'

# Fungsi untuk memuat model (dengan cache Streamlit)
@st.cache_resource
def load_models():
    # Pastikan data yang diproses ada
    if not os.path.exists(PROCESSED_DATA_DIR) or not os.listdir(PROCESSED_DATA_DIR):
        st.warning("Data yang diproses ('data/processed') tidak ditemukan. Menjalankan preprocessing...")
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.txt')]
        if not raw_files:
            st.error(f"Tidak ada file di {RAW_DATA_DIR}. Pastikan file data mentah ada.")
            return None, None
            
        for doc_id in raw_files:
            with open(os.path.join(RAW_DATA_DIR, doc_id), 'r', encoding='utf-8') as f_in:
                raw_content = f_in.read()
            processed_content = preprocess(raw_content)
            with open(os.path.join(PROCESSED_DATA_DIR, doc_id), 'w', encoding='utf-8') as f_out:
                f_out.write(processed_content)
        st.success("Preprocessing selesai.")

    vsm_model = VectorSpaceModel(RAW_DATA_DIR)
    bool_model = BooleanRetrieval(PROCESSED_DATA_DIR)
    return vsm_model, bool_model

st.title("Mesin Pencari Berita COVID-19 (UTS STKI)")
st.write("Project ini mengimplementasikan Boolean Retrieval dan Vector Space Model.")

try:
    vsm_model, bool_model = load_models()

    if vsm_model and bool_model:
        # --- Bagian Vector Space Model (VSM) ---
        st.header("Pencarian Peringkat (Vector Space Model)")
        # 
        vsm_query = st.text_input("Masukkan query VSM (misal: 'varian delta jakarta'):", key="vsm_query")
        k_value = st.slider("Jumlah hasil (Top-K):", min_value=1, max_value=5, value=3)
        
        if st.button("Cari (VSM)"):
            if vsm_query:
                results = vsm_model.search(vsm_query, k=k_value)
                st.subheader(f"Top {k_value} Hasil Pencarian VSM:")
                
                if not results:
                    st.warning("Tidak ada dokumen relevan yang ditemukan.")
                else:
                    # (Tampilkan tabel: doc_id | cosine | snippet)
                    for res in results:
                        st.markdown(f"**Dokumen:** `{res['doc_id']}` | **Skor:** `{res['score']:.4f}`")
                        st.info(f"**Snippet:** {res['snippet']}")
                        with st.expander("Lihat Teks Asli"):
                            doc_index = vsm_model.doc_ids.index(res['doc_id'])
                            st.text(vsm_model.raw_documents[doc_index])
            else:
                st.error("Silakan masukkan query VSM.")

        st.divider()

        # --- Bagian Boolean Retrieval ---
        st.header("Pencarian Boolean")
        # 
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
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.error("Pastikan file data 'data/raw' ada dan 'src' dapat diakses.")
