# Proyek UAS: Sistem Temu Kembali Informasi (STKI)
## Mini Search Engine (Boolean & Vector Space Model)

 Proyek ini adalah implementasi dari sistem temu kembali informasi (STKI) mini sebagai pemenuhan Ujian Tengah Semester (UAS) Ganjil 2025/2026. Sistem ini dibangun menggunakan Python, mampu mengindeks 5 dokumen berita, dan mendukung dua model pencarian: **Boolean Retrieval** dan **Vector Space Model (VSM)** dengan perankingan TF-IDF.

---

## 🧑‍🎓 Informasi Mahasiswa

* **Nama:** `Rizka Nugraha`
* **NIM:** `A11.2022.14119`
* **Mata Kuliah:** Sistem Temu Kembali Informasi (A11.4703)
* **Dosen:** Abu Salam, M.Kom
* **Universitas:** Universitas Dian Nuswantoro

---

## 🎯 Fitur Utama

Proyek ini mengimplementasikan semua komponen:

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

Struktur repositori ini mengikuti format yang ditentukan.
```
stki-uas/ 
├── app/ 
│ └── main.py
├── data/ 
│ ├── raw/
│ └── processed/  
├── notebooks/ 
│ └── STKI_UAS_A11.2022.14119_RIZKA_NUGRAHA.ipynb  
├── reports/ 
│ └── laporan_stki_rizka.pdf
├── src/ 
│ ├── init.py 
│ ├── preprocess.py 
│ ├── boolean_ir.py 
│ ├── vsm_ir.py 
│ ├── search.py 
│ └── eval.py 
├── readme.md 
└── requirements.txt 
```

---

## 🚀 Cara Menjalankan

### 1. Menjalankan di Google Colab (Pengembangan)

Cara ini digunakan untuk pengembangan dan pengujian, seperti yang terlihat pada file `.ipynb`.

1.  Upload proyek ini ke Google Drive Anda.
2.  Buka file `notebooks/STKI_UAS_A11.2022.14119_RIZKA_NUGRAHA.ipynb` di Google Colab.
3.  Jalankan sel `drive.mount()` untuk menghubungkan ke Google Drive.
4.  Jalankan sel-sel instalasi (`!pip install ...`).
5.  Jalankan sel-sel unduh NLTK (`nltk.download(...)`).
6.  Jalankan sel-sel pengujian (Soal 02, 03, 04, 05) secara berurutan. Sel terakhir akan meluncurkan aplikasi Streamlit menggunakan `pyngrok`.

### 2. Menjalankan Secara Lokal (Deployment)

Cara ini digunakan untuk menjalankan aplikasi Streamlit di komputer Anda.

1.  **Clone repositori:**
    ```bash
    git clone https://github.com/rizkanugrha/stki.git
    cd stki
    git checkout UAS

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
