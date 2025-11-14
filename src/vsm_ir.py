import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import preprocess

class VectorSpaceModel:
    # --- PERUBAHAN DI SINI ---
    # Tambahkan parameter 'sublinear_tf'
    def __init__(self, raw_data_dir, sublinear_tf=False):
        self.raw_data_dir = raw_data_dir
        self.doc_ids = []
        self.raw_documents = []
        self.processed_documents = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
        self._load_and_process_data()
        # Teruskan parameter ke _build_vsm
        self._build_vsm(use_sublinear=sublinear_tf)

    def _load_and_process_data(self):
        """Memuat dan memproses data dari direktori raw."""
        doc_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.endswith('.txt')])
        
        for doc_id in doc_files:
            self.doc_ids.append(doc_id)
            filepath = os.path.join(self.raw_data_dir, doc_id)
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_content = f.read()
                self.raw_documents.append(raw_content)
                processed_content = preprocess(raw_content)
                self.processed_documents.append(processed_content)

    # --- PERUBAHAN DI SINI ---
    def _build_vsm(self, use_sublinear=False):
        """Membangun matriks TF-IDF."""
        # Gunakan opsi sublinear_tf=True jika diminta
        self.vectorizer = TfidfVectorizer(sublinear_tf=use_sublinear)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_documents)
    
    def get_tfidf_matrix(self):
        """Mengembalikan matriks TF-IDF sebagai DataFrame."""
        terms = self.vectorizer.get_feature_names_out()
        df_tfidf = pd.DataFrame(self.tfidf_matrix.toarray(), index=self.doc_ids, columns=terms)
        return df_tfidf

    def search(self, query, k=5):
        """Mencari query menggunakan VSM dan mengembalikan top-k hasil."""
        processed_query = preprocess(query)
        query_vec = self.vectorizer.transform([processed_query])
        cosine_sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        sorted_indices = np.argsort(cosine_sims)[::-1]
        
        results = []
        for i in sorted_indices[:k]:
            if cosine_sims[i] > 0.01:
                doc_id = self.doc_ids[i]
                score = cosine_sims[i]
                snippet = self.raw_documents[i].replace("\n", " ").strip()
                snippet = (snippet[:120] + '...') if len(snippet) > 120 else snippet
                
                results.append({
                    "doc_id": doc_id,
                    "score": score,
                    "snippet": snippet
                })
        return results
