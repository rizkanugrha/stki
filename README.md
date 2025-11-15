# Proyek UTS: Sistem Temu Kembali Informasi (STKI)
## Mini Search Engine (Boolean & Vector Space Model)

 Proyek ini adalah implementasi dari sistem temu kembali informasi (STKI) mini sebagai pemenuhan Ujian Tengah Semester (UTS) Ganjil 2025/2026. Sistem ini dibangun menggunakan Python, mampu mengindeks 5 dokumen berita, dan mendukung dua model pencarian: **Boolean Retrieval** dan **Vector Space Model (VSM)** dengan perankingan TF-IDF.

Aplikasi ini juga di-deploy menggunakan Streamlit Community Cloud dan dapat diakses secara publik.

**Tautan Aplikasi Streamlit:** `https://uts-stki-14119.streamlit.app`

---

## 🧑‍🎓 Informasi Mahasiswa

* **Nama:** `Rizka Nugraha`
* **NIM:** `A11.2022.14119`
* **Mata Kuliah:** Sistem Temu Kembali Informasi (A11.4703)
* **Dosen:** Abu Salam, M.Kom
* **Universitas:** Universitas Dian Nuswantoro

---

## 🎯 Fitur Utama

Proyek ini mengimplementasikan semua komponen yang diminta dalam soal UTS, dari Soal 02 hingga Soal 05:

### 1. Soal 02: Document Preprocessing 
Pipeline preprocessing teks Bahasa Indonesia yang lengkap (`src/preprocess.py`) untuk membersihkan dan menstandarisasi korpus dokumen, mencakup:
* **Case Folding:** Mengubah semua teks menjadi huruf kecil.
* **Normalisasi:** Menghapus URL, angka, dan tanda baca.
* **Tokenisasi:** Memecah teks menjadi token/kata (menggunakan NLTK).
* **Stopword Removal:** Membuang kata-kata umum (misal: 'yang', 'di', 'dan') (menggunakan NLTK).
* **Stemming:** Mengubah kata ke bentuk dasarnya (misal: 'meningkat' -> 'tingkat') (menggunakan Sastrawi).

### 2. Soal 03: Boolean Retrieval Model 
Implementasi model pencarian boolean klasik (`src/boolean_ir.py`)  yang:
* Membangun **Inverted Index** sederhana dari korpus yang telah diproses.
* Mendukung parser query boolean yang mampu memproses operator **AND, OR, dan NOT**.
* Dievaluasi menggunakan metrik **Precision, Recall, dan F1-Score** terhadap *gold standard* manual.

### 3. Soal 04: Vector Space Model (VSM) & Ranking 
Implementasi model pencarian VSM (`src/vsm_ir.py`)  untuk pencarian berperingkat:
* Membangun matriks Dokumen-Term menggunakan pembobotan **TF-IDF*.
* Merepresentasikan query sebagai vektor TF-IDF.
* Melakukan perankingan dokumen berdasarkan **Cosine Similarity** antara query dan dokumen.
* Mengembalikan **Top-K** hasil pencarian.
* Dievaluasi menggunakan metrik **Precision@k** dan **MAP@k**.

### 4. Soal 05: Search Engine & Evaluasi 
Proyek *capstone* yang menyatukan semua modul dan melakukan evaluasi lanjutan:
* **Perbandingan Term Weighting:** Membandingkan dua skema pembobotan (TF-IDF standar vs. TF-IDF Sublinear) dan melaporkan dampaknya terhadap metrik MAP@k.
* **Search Engine Orchestrator:** Sebuah skrip CLI (`src/search.py`)  yang dapat menerima argumen `--model {boolean, vsm}` dan `--query "..."`.
* **Main Interface:** Aplikasi web interaktif (`app/main.py`)  yang dibuat dengan **Streamlit**, memungkinkan pengguna untuk beralih antara mode pencarian Boolean dan VSM.

---

## 🛠️ Teknologi yang Digunakan

* **Python 3**
* **Streamlit:** Untuk membangun antarmuka pengguna (UI) aplikasi web.
* **Scikit-learn:** Untuk `TfidfVectorizer` dan `cosine_similarity`.
* **NLTK:** Untuk tokenisasi dan daftar *stopwords*.
* **Sastrawi:** Untuk *stemming* Bahasa Indonesia.
* **Pandas:** Untuk analisis dan visualisasi tabel evaluasi.
* **Google Colab:** Untuk lingkungan pengembangan dan pengujian notebook.
* **GitHub:** Untuk *version control* dan *deployment*.

---

## 📂 Struktur Folder

Struktur repositori ini mengikuti format yang ditentukan dalam soal UTS.
```
stki-uts/ 
├── app/ 
│ └── main.py # Skrip utama aplikasi Streamlit  
├── data/ 
│ ├── raw/ # Berisi 5 dokumen .txt (corpus) 
│ └── processed/ # (Dibuat otomatis oleh skrip preprocessing) 
├── notebooks/ 
│ └── STKI_UTS_A11.2022.14119_RIZKA_NUGRAHA.ipynb # Notebook Colab untuk pengujian & dev  
├── reports/ 
│ └── (laporan.pdf) # (Placeholder untuk laporan PDF)  
├── src/ 
│ ├── init.py # Membuat 'src' menjadi Python package 
│ ├── preprocess.py # Modul Soal 02 (Preprocessing) 
│ ├── boolean_ir.py # Modul Soal 03 (Boolean Model) 
│ ├── vsm_ir.py # Modul Soal 04 (VSM) 
│ ├── search.py # Skrip CLI (Soal 05) 
│ └── eval.py # Modul evaluasi (P/R/F1, MAP)  
├── readme.md # File ini 
└── requirements.txt # Dependensi Python untuk deployment
```

---

## 🚀 Cara Menjalankan

### 1. Menjalankan di Google Colab (Pengembangan)

Cara ini digunakan untuk pengembangan dan pengujian, seperti yang terlihat pada file `.ipynb`.

1.  Upload proyek ini ke Google Drive Anda.
2.  Buka file `notebooks/STKI_UTS_A11.2022.14119_RIZKA_NUGRAHA.ipynb` di Google Colab.
3.  Jalankan sel `drive.mount()` untuk menghubungkan ke Google Drive.
4.  Jalankan sel `%cd` untuk pindah ke direktori root proyek (`.../UTS`).
5.  Jalankan sel-sel instalasi (`!pip install ...`).
6.  Jalankan sel-sel unduh NLTK (`nltk.download(...)`).
7.  Jalankan sel-sel pengujian (Soal 02, 03, 04, 05) secara berurutan. Sel terakhir akan meluncurkan aplikasi Streamlit menggunakan `pyngrok`.

### 2. Menjalankan Secara Lokal (Deployment)

Cara ini digunakan untuk menjalankan aplikasi Streamlit di komputer Anda.

1.  **Clone repositori:**
    ```bash
    git clone https://github.com/rizkanugrha/stki.git
    cd stki
    git checkout UTS

    ```

2.  **Buat virtual environment (disarankan):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # (Di Windows: venv\Scripts\activate)
    ```

3.  **Install dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Unduh data NLTK (hanya sekali):**
    ```bash
    python -m nltk.downloader punkt
    python -m nltk.downloader punkt_tab
    python -m nltk.downloader stopwords
    ```

5.  **Jalankan aplikasi Streamlit:**
    ```bash
    streamlit run app/main.py
    ```
    Aplikasi akan terbuka secara otomatis di browser Anda di `http://localhost:8501`.
