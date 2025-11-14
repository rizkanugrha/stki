
import streamlit as st
import os
import sys
import pandas as pd
import nltk  # 1. Import NLTK

# --- KONFIGURASI HALAMAN ---
# st.set_page_config() harus menjadi perintah Streamlit PERTAMA yang dijalankan.
st.set_page_config(
    page_title="Mesin Pencari STKI",
    page_icon="🔎",
    layout="wide"
)
# --------------------------------------------------------


# --- PERBAIKAN DEPLOYMENT STREAMLIT (Soal LookupError) ---
# Mengunduh data NLTK yang diperlukan
print("Memulai download data NLTK (stopwords & punkt)...")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
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

@st.cache_resource
def load_models():
    """
    Memuat semua model yang diperlukan.
    Fungsi ini DI-CACHE dan TIDAK BOLEH memanggil elemen UI Streamlit.
    """
    # Pastikan data yang diproses ada
    if not os.path.exists(PROCESSED_DATA_DIR) or not os.listdir(PROCESSED_DATA_DIR):
        # DIHAPUS: st.warning(...)
        print(f"Data yang diproses ('{PROCESSED_DATA_DIR}') tidak ditemukan. Menjalankan preprocessing...")
        
        if not os.path.exists(RAW_DATA_DIR):
            # DIHAPUS: st.error(...)
            print(f"FATAL: Folder {RAW_DATA_DIR} tidak ditemukan. Tidak bisa memuat data.")
            return None, None, None
            
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.txt')]
        
        if not raw_files:
            # DIHAPUS: st.error(...)
            print(f"Tidak ada file .txt di {RAW_DATA_DIR}.")
            return None, None, None
            
        # Lakukan preprocessing
        for doc_id in raw_files:
            with open(os.path.join(RAW_DATA_DIR, doc_id), 'r', encoding='utf-8') as f_in:
                raw_content = f_in.read()
            processed_content = preprocess(raw_content)
            with open(os.path.join(PROCESSED_DATA_DIR, doc_id), 'w', encoding='utf-8') as f_out:
                f_out.write(processed_content)
        # DIHAPUS: st.success(...)
        print("Preprocessing selesai. Model siap dimuat.")

    # Muat model
    # DIHAPUS: st.toast(...)
    print("Memuat model VSM (Default TF-IDF)...")
    vsm_model_default = VectorSpaceModel(RAW_DATA_DIR, sublinear_tf=False) 
    
    # DIHAPUS: st.toast(...)
    print("Memuat model VSM (Sublinear TF-IDF)...")
    vsm_model_sublinear = VectorSpaceModel(RAW_DATA_DIR, sublinear_tf=True)
    
    # DIHAPUS: st.toast(...)
    print("Memuat model Boolean...")
    bool_model = BooleanRetrieval(PROCESSED_DATA_DIR)
    
    # DIHAPUS: st.toast(...)
    print("Semua model berhasil dimuat!")
    return vsm_model_default, vsm_model_sublinear, bool_model
# --------------------------------------------------------


# --- Tampilan Utama Aplikasi Streamlit ---
st.title("Mesin Pencari Berita COVID-19 (UTS STKI)")
st.write("Nama: Rizka Nugraha \nNIM: A11.2022.14119\n")
st.write("Project ini mengimplementasikan Boolean Retrieval dan Vector Space Model.")

try:
    # Coba muat model
    vsm_model_default, vsm_model_sublinear, bool_model = load_models()

    # Periksa apakah model berhasil dimuat (di luar fungsi cache)
    if vsm_model_default and vsm_model_sublinear and bool_model:
        
        # Buat Tabs UI
        tab_search, tab_eval_vsm, tab_eval_bool = st.tabs([
            "🔎 Pencarian (VSM & Boolean)", 
            "📊 Evaluasi VSM (Soal 05)", 
            "🧮 Evaluasi Boolean (Soal 03)"
        ])
        
        # ==========================================================
        # TAB 1: PENCARIAN
        # ==========================================================
        with tab_search:
            st.header("Coba Mesin Pencari")
            
            # --- Bagian Vector Space Model (VSM) ---
            st.subheader("Pencarian Peringkat (Vector Space Model)")
            st.info("Menggunakan model VSM dengan **Sublinear TF-IDF** (Skema terbaik).")
            vsm_query = st.text_input("Masukkan query VSM (misal: 'varian delta jakarta'):", key="vsm_query")
            k_value = st.slider("Jumlah hasil (Top-K):", min_value=1, max_value=5, value=3)
            
            if st.button("Cari (VSM)"):
                if vsm_query:
                    # Gunakan model sublinear untuk pencarian demo
                    results = vsm_model_sublinear.search(vsm_query, k=k_value)
                    st.subheader(f"Top {k_value} Hasil Pencarian VSM:")
                    if not results:
                        st.warning("Tidak ada dokumen relevan yang ditemukan (skor > 0.01).")
                    else:
                        for res in results:
                            st.markdown(f"**Dokumen:** `{res['doc_id']}` | **Skor:** `{res['score']:.4f}`")
                            st.info(f"**Snippet:** {res['snippet']}")
                else:
                    st.error("Silakan masukkan query VSM.")

            st.divider()

            # --- Bagian Boolean Retrieval ---
            st.subheader("Pencarian Boolean")
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
                else:
                    st.error("Silakan masukkan query Boolean.")
        
        # ==========================================================
        # TAB 2: EVALUASI VSM (SOAL 05)
        # ==========================================================
        with tab_eval_vsm:
            st.header("Perbandingan Skema VSM (Soal 05)")
            st.markdown("Evaluasi ini membandingkan **Default TF-IDF** dengan **Sublinear TF-IDF** menggunakan metrik P@3 dan MAP@5.")
            
            # Gold standard VSM (dari notebook)
            gold_standard_vsm = {
                "vaksin AND delta": {"berita2.txt", "berita3.txt"},
                "ppkm OR jakarta": {"berita1.txt", "berita4.txt"},
                "kasus NOT amerika": {"berita1.txt", "berita4.txt"} 
            }
            map_k = 5
            p_k = 3
            
            # Jalankan evaluasi
            eval_data_pak = []
            query_results_default = {}
            query_results_sublinear = {}
            
            for query, relevant_docs in gold_standard_vsm.items():
                # 1. Model Default
                res_default = vsm_model_default.search(query, k=map_k)
                retrieved_default = [r['doc_id'] for r in res_default]
                p_at_k_default = precision_at_k(retrieved_default, relevant_docs, k=p_k)
                query_results_default[query] = retrieved_default

                # 2. Model Sublinear
                res_sublinear = vsm_model_sublinear.search(query, k=map_k)
                retrieved_sublinear = [r['doc_id'] for r in res_sublinear]
                p_at_k_sublinear = precision_at_k(retrieved_sublinear, relevant_docs, k=p_k)
                query_results_sublinear[query] = retrieved_sublinear
                
                eval_data_pak.append({
                    "Query": query,
                    f"P@{p_k} (Default)": p_at_k_default,
                    f"P@{p_k} (Sublinear)": p_at_k_sublinear
                })

            # Hitung MAP
            map_default = mean_average_precision(query_results_default, gold_standard_vsm, k=map_k)
            map_sublinear = mean_average_precision(query_results_sublinear, gold_standard_vsm, k=map_k)

            st.subheader(f"Perbandingan MAP@{map_k} (Skor Keseluruhan)")
            df_map = pd.DataFrame([
                {"Skema": "Default TF-IDF", f"MAP@{map_k}": map_default},
                {"Skema": "Sublinear TF-IDF", f"MAP@{map_k}": map_sublinear}
            ]).set_index("Skema")
            st.dataframe(df_map, use_container_width=True)
            
            st.subheader(f"Perbandingan Precision@{p_k} (Per Kueri)")
            df_pak = pd.DataFrame(eval_data_pak).set_index("Query")
            st.dataframe(df_pak, use_container_width=True)
            
            with st.expander("Analisis Singkat"):
                st.write(f"""
                - **MAP@{map_k}** (Mean Average Precision) mengukur kualitas perankingan secara keseluruhan.
                - Pada kasus ini, model **{"Sublinear" if map_sublinear >= map_default else "Default"}** memiliki skor MAP yang sedikit lebih baik.
                - Sublinear TF (`1 + log(tf)`) berguna untuk menangani frekuensi kata yang sangat tinggi, sehingga kata yang muncul 1000x tidak dianggap 1000x lebih penting daripada yang muncul 1x.
                """)
            with st.expander("Lihat Hasil Retrieval Mentah (untuk Debug)"):
                st.json({
                    "Default TF-IDF": query_results_default,
                    "Sublinear TF-IDF": query_results_sublinear,
                    "Gold Standard": {q: list(d) for q, d in gold_standard_vsm.items()}
                })

        # ==========================================================
        # TAB 3: EVALUASI BOOLEAN (SOAL 03)
        # ==========================================================
        with tab_eval_bool:
            st.header("Evaluasi Model Boolean (Soal 03)")
            st.markdown("Evaluasi ini menjalankan 3 kueri (`AND`, `OR`, `NOT`) terhadap *gold standard* manual.")
            
            # Gold standard Boolean (dari notebook)
            gold_standard_bool = {
                "vaksin AND delta": {"berita3.txt"},
                "ppkm OR jakarta": {"berita1.txt", "berita4.txt"},
                "kasus NOT amerika": {"berita1.txt", "berita2.txt", "berita3.txt", "berita4.txt"}
            }
            
            eval_results = []
            explain_logs = {}

            for query, relevant_docs in gold_standard_bool.items():
                retrieved_docs, explain = bool_model.process_query(query)
                p, r, f1 = calculate_precision_recall_f1(retrieved_docs, relevant_docs)
                
                eval_results.append({
                    "Query": query,
                    "Retrieved Docs (Hasil)": ", ".join(retrieved_docs) if retrieved_docs else "None",
                    "Relevant Docs (Kunci)": ", ".join(relevant_docs),
                    "Precision": p,
                    "Recall": r,
                    "F1-Score": f1
                })
                explain_logs[query] = explain
            
            st.subheader("Hasil Evaluasi (Precision, Recall, F1)")
            df_eval_bool = pd.DataFrame(eval_results).set_index("Query")
            st.dataframe(df_eval_bool, use_container_width=True)

            with st.expander("Lihat Penjelasan (Explain) Eksekusi Kueri"):
                st.json(explain_logs)
    
    # Jika model gagal dimuat
    else:
        st.error("Model gagal dimuat. Periksa log Streamlit untuk detailnya.")
        st.error("Pastikan file data 'data/raw' ada dan 'src' dapat diakses.")

except Exception as e:
    # Tangkap error umum (seperti import gagal, dll)
    st.error(f"Terjadi kesalahan fatal saat aplikasi dimulai: {e}")
    st.error("Pastikan file data 'data/raw' ada dan 'src' dapat diakses.")
    st.code(f"Detail error: {e}", language="text")